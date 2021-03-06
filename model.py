import chess
import ranger_adabelief
import ranger
from Lookahead import SGD, GradualWarmupScheduler
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau
from adabound import AdaBoundW
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl

# 3 layer fully connected network
L1 = 256
L2 = 32
L3 = 32

class NNUE(pl.LightningModule):
  """
  This model attempts to directly represent the nodchip Stockfish trainer methodology.

  lambda_ = 0.0 - purely based on game results
  lambda_ = 1.0 - purely based on search scores

  It is not ideal for training a Pytorch quantized model directly.
  """
  def __init__(self, feature_set, lambda_=1.0, learning_rate=1e-3, batch_per_epoch=None, config=None):
    super(NNUE, self).__init__()

    self.input = nn.Linear(feature_set.num_features, L1)
    self.feature_set = feature_set
    self.l1 = nn.Linear(2 * L1, L2)
    self.l2 = nn.Linear(L2, L3)
    self.output = nn.Linear(L3, 1)
    self.lambda_ = lambda_

    self.eps = 1e-16
    self.weight_decay = 0
    self.hparams.learning_rate = learning_rate
    self.batch_per_epoch = batch_per_epoch

    if config is not None:
      self.eps = config["eps"]
      self.weight_decay = config["weight_decay"]
      self.hparams.learning_rate = config["lr"]

    self.save_hyperparameters()

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

    w = self.input(w_in)
    b = self.input(b_in)
    l0_ = (us * torch.cat([w, b], dim=1)) + (them * torch.cat([b, w], dim=1))
    # clamp here is used as a clipped relu to (0.0, 1.0)
    l0_ = torch.Clamp(l0_, 0.0, 1.0)
    l1_ = torch.clamp(self.l1(l0_), 0.0, 1.0)
    l2_ = torch.clamp(self.l2(l1_), 0.0, 1.0)
    x = self.output(l2_)
    return x

  def step_(self, batch, batch_idx, loss_type):
    us, them, white, black, outcome, score = batch
    q = self(us, them, white, black)
    t = outcome
    # Divide score by 600.0 to match the expected NNUE scaling factor
    p = (score / 600.0).sigmoid()
    epsilon = 1e-12
    teacher_entropy = -(p * (p + epsilon).log() + (1.0 - p) * (1.0 - p + epsilon).log())
    outcome_entropy = -(t * (t + epsilon).log() + (1.0 - t) * (1.0 - t + epsilon).log())
    teacher_loss = -(p * F.logsigmoid(q) + (1.0 - p) * F.logsigmoid(-q))
    outcome_loss = -(t * F.logsigmoid(q) + (1.0 - t) * F.logsigmoid(-q))
    result  = self.lambda_ * teacher_loss    + (1.0 - self.lambda_) * outcome_loss
    entropy = self.lambda_ * teacher_entropy + (1.0 - self.lambda_) * outcome_entropy
    loss = result.mean() - entropy.mean()
    return loss


    # MSE Loss function for debugging
    # Scale score by 600.0 to match the expected NNUE scaling factor
    # output = self(us, them, white, black) * 600.0
    # loss = F.mse_loss(output, score)

  def training_step(self, batch, batch_idx):
    loss = self.step_(batch, batch_idx, 'train_loss')
    self.log('train_loss',loss)
    return loss

  def validation_step(self, batch, batch_idx):
    loss = self.step_(batch, batch_idx, 'val_loss')
    self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
    return {"val_loss": loss}

  # def validation_epoch_end(self, outputs):
  #   avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
  #   self.log("ptl/val_loss", avg_loss)

  def test_step(self, batch, batch_idx):
    return self.step_(batch, batch_idx, 'test_loss')

  def configure_optimizers(self):
    #optimizer = ranger.Ranger(self.parameters())
    #optimizer = AdaBoundW(self.parameters(), epochs=100, steps_per_epoch=self.batch_per_epoch,
                          #weight_decay=0)
    optimizer = ranger_adabelief.RangerAdaBelief(self.parameters(), lr=self.hparams.learning_rate, eps=self.eps,
                                                 betas=(0.9, 0.999), weight_decay=self.weight_decay)
    #optimizer = SGD(self.parameters(), lr=self.hparams.learning_rate, momentum=0.9, weight_decay=0,
                    #use_gc=True, k=5, alpha=0.5)
    #scheduler = OneCycleLR(optimizer, max_lr=0.00005, steps_per_epoch=self.batch_per_epoch, epochs=50)

    # lr_scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=10, verbose=True)
    #
    # scheduler = {
    #   'scheduler': lr_scheduler,
    #   'reduce_on_plateau': True,
    #  # val_checkpoint_on is val_loss passed in as checkpoint_on
    #   'monitor': 'val_loss'
    # }


    return optimizer#], [scheduler]
