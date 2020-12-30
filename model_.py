import chess
import torch
from torch import nn
import torch.nn.functional as F
from torch.quantization import QuantStub, DeQuantStub
from torch.nn.quantized import FloatFunctional
# 3 layer fully connected network


class NNUE(nn.Module):
  """
  This model attempts to directly represent the nodchip Stockfish trainer methodology.

  lambda_ = 0.0 - purely based on game results
  lambda_ = 1.0 - purely based on search scores

  It is not ideal for training a Pytorch quantized model directly.
  """
  def __init__(self, feature_set, lambda_=1.0, s=1):
    super(NNUE, self).__init__()

    L1 = 256 * s
    L2 = 32 * s
    L3 = 32 * s

    self.input = nn.Linear(feature_set.num_features, L1)
    self.R0 = nn.ReLU()
    self.feature_set = feature_set
    self.l1 = nn.Linear(2 * L1, L2)
    self.R1 = nn.ReLU()
    self.l2 = nn.Linear(L2, L3)
    self.R2 = nn.ReLU()
    self.output = nn.Linear(L3, 1)
    self.lambda_ = lambda_

    self.quant = QuantStub()
    self.dequant = DeQuantStub()

    self.input_mul = FloatFunctional()
    self.input_add = FloatFunctional()

    self._zero_virtual_feature_weights()

  '''
  We zero all virtual feature weights because during serialization to .nnue
  we compute weights for each real feature as being the sum of the weights for
  the real feature in question and the virtual features it can be factored to.
  This means that if we didn't initialize the virtual feature weights to zero
  we would end up with the real features having effectively unexpected values
  at initialization - following the bell curve based on how many factors there are.
  '''
  def _zero_virtual_feature_weights(self):
    weights = self.input.weight
    for a, b in self.feature_set.get_virtual_feature_ranges():
      weights[a:b, :] = 0.0
    self.input.weight = nn.Parameter(weights)

  '''
  This method attempts to convert the model from using the self.feature_set
  to new_feature_set.
  '''
  def set_feature_set(self, new_feature_set):
    if self.feature_set.name == new_feature_set.name:
      return

    # TODO: Implement this for more complicated conversions.
    #       Currently we support only a single feature block.
    if len(self.feature_set.features) > 1:
      raise Exception('Cannot change feature set from {} to {}.'.format(self.feature_set.name, new_feature_set.name))

    # Currently we only support conversion for feature sets with
    # one feature block each so we'll dig the feature blocks directly
    # and forget about the set.
    old_feature_block = self.feature_set.features[0]
    new_feature_block = new_feature_set.features[0]

    # next(iter(new_feature_block.factors)) is the way to get the
    # first item in a OrderedDict. (the ordered dict being str : int
    # mapping of the factor name to its size).
    # It is our new_feature_factor_name.
    # For example old_feature_block.name == "HalfKP"
    # and new_feature_factor_name == "HalfKP^"
    # We assume here that the "^" denotes factorized feature block
    # and we would like feature block implementers to follow this convention.
    # So if our current feature_set matches the first factor in the new_feature_set
    # we only have to add the virtual feature on top of the already existing real ones.
    if old_feature_block.name == next(iter(new_feature_block.factors)):
      # We can just extend with zeros since it's unfactorized -> factorized
      weights = self.input.weight
      padding = weights.new_zeros((weights.shape[0], new_feature_block.num_virtual_features))
      weights = torch.cat([weights, padding], dim=1)
      self.input.weight = nn.Parameter(weights)
      self.feature_set = new_feature_set
    else:
      raise Exception('Cannot change feature set from {} to {}.'.format(self.feature_set.name, new_feature_set.name))

  def forward(self, us, them, w_in, b_in):

    us = self.quant(us)
    them = self.quant(them)
    w_in = self.quant(w_in)
    b_in = self.quant(b_in)

    w = self.input(w_in)
    b = self.input(b_in)
    l0_ = self.input_add.add(self.input_mul.mul(us, torch.cat([w, b], dim=1)),
                             self.input_mul.mul(them, torch.cat([b, w], dim=1)))

    l0_ = self.R0(l0_)
    l1_ = self.R1(self.l1(l0_))
    l2_ = self.R2(self.l2(l1_))
    x = self.output(l2_)

    x = self.dequant(x)

    return x

  def get_1xlr(self):

    list = [module for module in self.children()]

    for i in list:
        if i == self.input:
          if isinstance(i, nn.Linear):
            for p in i.parameters():
              if p.requires_grad:
                yield p

  def get_10xlr(self):

    list = [module for module in self.children()]

    for i in list:
        if i != self.input:
          if isinstance(i, nn.Linear):
            for p in i.parameters():
              if p.requires_grad:
                yield p

