class NNUEQuantizedWriter():
  """
  All values are stored in little endian.
  """
  def __init__(self, model):
    self.buf = bytearray()

    for m in model.modules():
      print(m)

    self.write_header()
    self.int32(0x5d69d7b8) # Feature transformer hash
    self.write_feature_transformer(model.input)
    self.int32(0x63337156) # FC layers hash
    self.write_fc_layer(model.l1)
    self.write_fc_layer(model.l2)
    self.write_fc_layer(model.output, is_output=True)

  def write_header(self):
    self.int32(0x7AF32F17) # version
    self.int32(0x3e5aa6ee) # halfkp network hash
    description = b"Features=HalfKP(Friend)[41024->256x2],"
    description += b"Network=AffineTransform[1<-32](ClippedReLU[32](AffineTransform[32<-32]"
    description += b"(ClippedReLU[32](AffineTransform[32<-512](InputSlice[512(0:512)])))))"
    self.int32(len(description)) # Network definition
    self.buf.extend(description)

  def write_feature_transformer(self, layer):
    # int16 bias
    # int16 weight
    bias = layer.bias().data
    self.buf.extend(bias.flatten().numpy().astype(numpy.int16).tobytes())
    weight = layer.weight().data
    # weights stored as [41024][256], so we need to transpose the pytorch [256][41024]
    self.buf.extend(weight.transpose(0, 1).flatten().numpy().astype(numpy.int16).tobytes())

  def write_fc_layer(self, layer, is_output=False):
    # FC layers are stored as int8 weights, and int32 biases
    bias = layer.bias().data
    self.buf.extend(bias.flatten().numpy().astype(numpy.int32).tobytes())
    weight = layer.weight().data
    # Stored as [outputs][inputs], so we can flatten
    self.buf.extend(weight.flatten().numpy().tobytes())

  def int16(self, v):
    self.buf.extend(struct.pack("<h", v))

  def int32(self, v):
    self.buf.extend(struct.pack("<i", v))