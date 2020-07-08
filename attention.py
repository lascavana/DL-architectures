import tensorflow.keras as K
import tensorflow as tf

class MultiHeadAttentionLayer(K.layers.Layer):
    def __init__(self, num_heads, depth, d_q, d_v=None):
        super(MultiHeadAttentionLayer, self).__init__()

        self.num_heads = num_heads
        self.depth = depth
        self.units = depth * num_heads

        self.q_layer = K.layers.Dense(self.units)
        self.k_layer = K.layers.Dense(self.units)
        self.v_layer = K.layers.Dense(self.units)
        self.linear_layer = K.layers.Dense(depth, activation='relu')

        # Build right away
        if d_v == None: d_v = d_q
        self.build((d_q,d_v))

    def build(self,input_shapes):
        d_q, d_v = input_shapes
        if not self.built:
            self.q_layer.build(d_q)
            self.k_layer.build(d_q)
            self.v_layer.build(d_v)
            self.linear_layer.build(self.units)
            self.built = True

    def split(self, x):
        # Split last dimension into heads
        shape = list(x.shape)[:-1] + [self.num_heads,self.depth]
        x = tf.reshape(x, tuple(shape))
        x = self.transpose(x)
        return x

    def transpose(self,x):
        # Transpose dimensions -2 and -3
        shape = list(x.shape)
        perm = [*range(len(shape))]
        perm[-2] = len(shape)-3
        perm[-3] = len(shape)-2
        return tf.transpose(x, perm=perm)

    def get_attention_mask(self,mask):
        """ Mask must be a binary tensor of dimensions
            [batch_size, ..., num_queries]
            indicating which queries to mask out. """
        n = mask.shape[-1]
        mask = tf.expand_dims(mask,axis=-1)
        mask = tf.repeat(mask,[n],axis=-1)

        # Create transposed tensor
        shape = list(mask.shape)
        perm = [*range(len(shape))]
        perm[-1] = len(shape)-2
        perm[-2] = len(shape)-1
        mask_t = tf.transpose(mask,perm=perm)

        return tf.multiply(mask,mask_t)

    def get_output_mask(self,mask):
        mask = tf.expand_dims(mask,axis=-1)
        mask = tf.repeat(mask,[self.depth],axis=-1)
        return mask

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        qk = tf.matmul(Q, K, transpose_b=True)
        dk = tf.cast(tf.shape(K)[-1], tf.float32)
        scaled_attention = qk / tf.math.sqrt(dk)

        if mask is not None:
            mask = get_attention_mask(mask)
            scaled_attention += (mask * -1e9)

        weights = tf.nn.softmax(scaled_attention, axis=-1)
        output = tf.matmul(weights, V)

        return output, weights

    def call(self, Q, K=None, V=None, mask=None):
        if K == None and V==None:
            # Self attention
            K = Q
            V = Q

        # Run through linear layers
        Q = self.q_layer(Q)
        K = self.k_layer(K)
        V = self.v_layer(V)

        # Split the heads
        Q = self.split(Q)
        K = self.split(K)
        V = self.split(V)

        # Run through attention
        attention_output, weights = self.scaled_dot_product_attention(Q, K, V, mask)

        # Prepare for the rest of processing
        attention_output= self.transpose(attention_output)
        shape = list(attention_output.shape)[:-2] + [self.units]
        concat_attention = tf.reshape(attention_output, tuple(shape))

        # Run through final linear layer
        output = self.linear_layer(concat_attention)
        if mask is not None:
            output = tf.multiply(output,self.get_output_mask(mask))

        return output #, weights
