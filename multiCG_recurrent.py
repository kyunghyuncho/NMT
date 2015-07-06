import numpy

from blocks.bricks import (Tanh, Linear,
                           Initializable, MLP, Logistic)
from blocks.bricks.base import application
from blocks.bricks.recurrent import recurrent, Bidirectional, BaseRecurrent
from blocks.roles import add_role, WEIGHT, INITIAL_STATE
from blocks.utils import shared_floatx_nans, shared_floatx_zeros

from picklable_itertools.extras import equizip
from theano import tensor


class BidirectionalWMT15(Bidirectional):
    """Wrapper to use two RNNs with separate word embedding matrices."""

    @application
    def apply(self, forward_dict, backward_dict):
        """Applies forward and backward networks and concatenates outputs."""
        forward = self.children[0].apply(as_list=True, **forward_dict)
        backward = [x[::-1] for x in
                    self.children[1].apply(reverse=True, as_list=True,
                                           **backward_dict)]
        return [tensor.concatenate([f, b], axis=2)
                for f, b in equizip(forward, backward)]


class GRUwithContext(BaseRecurrent, Initializable):
    """Gated Recurrent Unit that conditions on multiple contexts.

    GRU that conditions not only input but also source selector.
    Source selector is separately embedded for input, reset and update gates.

    Parameters
    ----------
    attended_dim : int
        The reprentation dimension of state below (encoder).
    dim : int
        The dimension of the hidden state.
    context_dim : int
        The dimension of source selector, also equal to the number of encoders
        if multiple encoders are employed.
    activation : :class:`.Brick`
        The brick to apply as activation.

    Notes
    -----
    Initial state conditions on a concatenated representation of source
    selector and last hidden state of the encoders backward rnn. An MLP with
    tanh activation is applied to the concatenated representation to obtain
    initial state of GRU.

    TODO: Computation of attended embedders should be carried outside of scan
          step function for speed up.

    """
    def __init__(self, attended_dim, dim, context_dim, activation=None,
                 gate_activation=None, **kwargs):
        super(GRUwithContext, self).__init__(**kwargs)
        self.dim = dim

        if not activation:
            activation = Tanh()
        if not gate_activation:
            gate_activation = Logistic()
        self.activation = activation
        self.gate_activation = gate_activation

        self.children = [activation, gate_activation]

        self.attended_dim = attended_dim
        self.context_dim = context_dim

        # Transformer for initial state
        self.initial_transformer = MLP(
                activations=[Tanh()],
                dims=[attended_dim + context_dim, self.dim],
                name='state_initializer')
        self.children.append(self.initial_transformer)

        # Gate transformers for source selector
        self.src_selector_gate_embedder = Linear(
                input_dim=context_dim,
                output_dim=self.dim * 2,
                use_bias=False,
                name='src_selector_gate_embedder')
        self.children.append(self.src_selector_gate_embedder)
        self.src_selector_input_embedder = Linear(
                input_dim=context_dim,
                output_dim=self.dim,
                use_bias=False,
                name='src_selector_input_embedder')
        self.children.append(self.src_selector_input_embedder)

    @property
    def state_to_state(self):
        return self.parameters[0]

    @property
    def state_to_gates(self):
        return self.parameters[1]

    def get_dim(self, name):
        if name == 'mask':
            return 0
        if name == 'gate_inputs':
            return 2 * self.dim
        if name in self.apply.sequences + self.apply.states:
            return self.dim
        if name in self.apply.contexts:
            return self.context_dim
        return super(GRUwithContext, self).get_dim(name)

    def _allocate(self):
        self.parameters.append(shared_floatx_nans((self.dim, self.dim),
                               name='state_to_state'))
        self.parameters.append(shared_floatx_nans((self.dim, 2 * self.dim),
                               name='state_to_gates'))
        #self.parameters.append(shared_floatx_zeros((self.dim,),
        #                       name="initial_state"))
        for i in range(2):
            if self.parameters[i]:
                add_role(self.parameters[i], WEIGHT)
        #add_role(self.parameters[2], INITIAL_STATE)

    def _initialize(self):
        self.weights_init.initialize(self.state_to_state, self.rng)
        state_to_update = self.weights_init.generate(
            self.rng, (self.dim, self.dim))
        state_to_reset = self.weights_init.generate(
            self.rng, (self.dim, self.dim))
        self.state_to_gates.set_value(
            numpy.hstack([state_to_update, state_to_reset]))

    @recurrent(sequences=['mask', 'inputs', 'gate_inputs'],
               states=['states'], outputs=['states'], contexts=['attended_1'])
    def apply(self, inputs, gate_inputs, states, mask=None, attended_1=None):
        """Apply the gated recurrent transition.

        Parameters
        ----------
        states : :class:`~tensor.TensorVariable`
            The 2 dimensional matrix of current states in the shape
            (batch_size, dim). Required for `one_step` usage.
        inputs : :class:`~tensor.TensorVariable`
            The 2 dimensional matrix of inputs in the shape (batch_size,
            dim)
        gate_inputs : :class:`~tensor.TensorVariable`
            The 2 dimensional matrix of inputs to the gates in the
            shape (batch_size, 2 * dim).
        mask : :class:`~tensor.TensorVariable`
            A 1D binary array in the shape (batch,) which is 1 if there is
            data available, 0 if not. Assumed to be 1-s only if not given.
        attended_1 : :class:`~tensor.TensorVariable`
            A 2 dimensional matrix of inputs to gates from attended 1st in the
            shape (batchs_size, attended_dim).


        Returns
        -------
        output : :class:`~tensor.TensorVariable`
            Next states of the network.

        """

        # TODO: move these computations out
        gate_src_embeds = self.src_selector_gate_embedder.apply(attended_1)
        input_src_embeds = self.src_selector_input_embedder.apply(attended_1)

        gate_values = self.gate_activation.apply(
            states.dot(self.state_to_gates) + gate_inputs + gate_src_embeds)
        update_values = gate_values[:, :self.dim]
        reset_values = gate_values[:, self.dim:]
        states_reset = states * reset_values
        next_states = self.activation.apply(
            states_reset.dot(self.state_to_state) + inputs + input_src_embeds)
        next_states = (next_states * update_values +
                       states * (1 - update_values))
        if mask:
            next_states = (mask[:, None] * next_states +
                           (1 - mask[:, None]) * states)
        return next_states

    @application(outputs=apply.states)
    def initial_states(self, batch_size, *args, **kwargs):
        """Conditions on last hidden state and source selector."""
        attended_0 = kwargs['attended_0']
        attended_1 = kwargs['attended_1']
        attended = tensor.concatenate(
            [attended_1, attended_0[0, :, -self.attended_dim:]],
            axis=1)
        initial_state = self.initial_transformer.apply(attended)
        return [initial_state]

    @apply.property('sequences')
    def apply_inputs(self):
        sequences = ['mask', 'inputs', 'gate_inputs']
        return sequences

    @apply.property('contexts')
    def apply_contexts(self):
        return ['attended_1']
