### Implementation of ResNet

# import the libraries
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import AveragePooling2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import ZeroPadding2D
from keras.layers.core import Activation
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import add
from keras.models import Model
from keras.regularizers import l2
from keras import backend as K


class ResNet:
	@staticmethod
	def residual_module(data, K, stride, chanDim, red=False, reg=0.0001, bnEps=2e-5, bnMom=0.9):
		"""
		data: input to the residual module
		K: number of filters that will be learned by the final CONV layer (the first two CONV layers will learn K/4 filters)
		stride: controls the stride of the convolution (will help us reduce spatial dimensions without using max pooling)
		chanDim: defines the axis which will perform batch normalization
		red: (reduce) will control whether we are reducing spatial dimensions (True) or not (False) as not all residual modules will reduce dimensions of our spatial volume
		reg: applies regularization strength for all CONV layers in the residual module
		bnEps: controls the Ɛ responsible for avoiding “division by zero” errors when normalizing inputs
		bnMom: controls the momentum for the moving average
		"""
		# the shortcut branch of the ResNet module should be initialize as the input (identity) data
		shortcut = data

		# the first block of the ResNet module are the 1x1 CONVs
		# BN ==> ReLu ==> CONV layer ==> pattern
		bn1 = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(data)
		act1 = Activation('relu')(bn1)
		conv1 = Conv2D(int(K * 0.25), (1, 1), use_bias=False, kernel_regularizer=l2(reg))(act1)

		# the second block of the ResNet module are the 3x3 CONVs
		bn2 = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(conv1)
		act2 = Activation('relu')(bn2)
		conv2 = Conv2D(int(K * 0.25), (3, 3), strides=stride, padding='same', use_bias=False,
				kernel_regularizer=l2(reg))(act2)

		# the third block of the ResNet module are 1x1 CONVs
		# we increase dims applying K filters with dims 1x1
		bn3 = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(conv2)
		act3 = Activation('relu')(bn3)
		conv3 = Conv2D(K, (1, 1), use_bias=False, kernel_regularizer=l2(reg))(act3)

		# to avoid applying max pooling, we need to check if reducing spatial dimensions is necessary
		# if we are to reduce the spatial size, apply CONV layer to the shortcut
		if red:
			shortcut = Conv2D(K, (1, 1), strides=stride, use_bias=False, kernel_regularizer=l2(reg))(act1)

		# add together the shortcut and the final CONV
		x = add([conv3, shortcut])

		# return the addition as the output of the ResNet module
		return x

	@staticmethod
	def build(width, height, depth, classes, stages, filters, reg=0.0001, bnEps=2e-5, bnMom=0.9):
		# initialize the input shape to be "channels last" and the channels dim itself
		inputShape = (height,width, depth)
		chanDim = -1

		# if we are using "channels first", update the input shape and the channels dim
		if K.image_data_format() == "channels first":
			inputShape = (depth, height, width)
			chanDim = 1

		# set the input and apply BN
		inputs = Input(shape=inputShape)
		x = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(inputs)

		# apply CONV ==> BN ==> ReLu ==> POOL to reduce spatial size
		x = Conv2D(filters[0], (5, 5), use_bias=False, padding='same', kernel_regularizer=l2(reg))(x)
		x = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(x)
		x = Activation('relu')(x)
		x = ZeroPadding2D((1, 1))(x)
		x = MaxPooling2D((3, 3), strides=(2, 2))(x)

		# loop over the number of stages
		for i in range(0,len(stages)):
			# initialize the stride, then apply residual module
			# used to reduce spatial size of the input volume
			stride = (1, 1) if i == 0 else (2, 2)
			x = ResNet.residual_module(x, filters[i + 1], stride, chanDim, red=True,
									bnEps=bnEps, bnMom=bnMom)

			# loop over the number of layers in each stage
			for j in range(0, stages[i] - 1):
				# apply ResNet module
				x = ResNet.residual_module(x, filters[i + 1], (1, 1), chanDim, bnEps=bnEps, bnMom=bnMom)

		# apply BN ==> ReLu ==> POOL
		x = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(x)
		x = Activation("relu")(x)
		x = AveragePooling2D((8, 8))(x)
		# softmax classifier
		x = Flatten()(x)
		x = Dense(classes, kernel_regularizer=l2(reg))(x)
		x = Activation("softmax")(x)
		# create the model
		model = Model(inputs, x, name="resnet")
		# return the constructed network architecture
		return model

