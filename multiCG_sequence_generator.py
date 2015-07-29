from blocks.bricks.base import application
from blocks.bricks.parallel import Fork
from blocks.bricks.sequence_generators import (
    BaseSequenceGenerator, LookupFeedback)
from blocks.utils import dict_union, dict_subset

from multiCG_attention import AttentionRecurrentWithMultiContext

from theano import tensor


class LookupFeedbackWMT15(LookupFeedback):
    """Feedback extension to zero out initial feedback.

    This brick extends LookupFeedback and overwrites its feedback method in
    order to provide all zeros as initial feedback for Groundhog compatibility.
    It may not be needed at all since learning BOS token is a cleaner and
    better option in sequences.

    """

    @application
    def feedback(self, outputs):
        assert self.output_dim == 0

        shp = [outputs.shape[i] for i in xrange(outputs.ndim)]
        outputs_flat = outputs.flatten()
        outputs_flat_zeros = tensor.switch(outputs_flat < 0, 0,
                                           outputs_flat)

        lookup_flat = tensor.switch(
            outputs_flat[:, None] < 0,
            tensor.alloc(0., outputs_flat.shape[0], self.feedback_dim),
            self.lookup.apply(outputs_flat_zeros))
        lookup = lookup_flat.reshape(shp+[self.feedback_dim])
        return lookup


class SequenceGeneratorWithMultiContext(BaseSequenceGenerator):
    """Sequence Generator that uses multiple contexts.

    The reason why we have such a generator is that the Sequence Generator
    structure in Blocks is not parametrized by its inner transition block.
    This sequence generator is only made in order to use
    AttentionRecurrentWithMultiContext.

    """
    def __init__(self, num_contexts, readout, transition, attention=None,
                 add_contexts=True, **kwargs):
        normal_inputs = [name for name in transition.apply.sequences
                         if 'mask' not in name]
        kwargs.setdefault('fork', Fork(normal_inputs))
        self.src_selector_fork = Fork(
            ['input_src_selector', 'update_src_selector', 'reset_src_selector'])
        self.num_contexts = num_contexts
        transition = AttentionRecurrentWithMultiContext(
            num_contexts, transition, attention,
            name="att_trans")
        super(SequenceGeneratorWithMultiContext, self).__init__(
            readout, transition, **kwargs)
        self.children.append(self.src_selector_fork)

    def _push_allocation_config(self):
        # Configure readout. That involves `get_dim` requests
        # to the transition. To make sure that it answers
        # correctly we should finish its configuration first.
        self.transition.push_allocation_config()
        transition_sources = (self._state_names + self._context_names +
                              self._glimpse_names)
        self.readout.source_dims = [self.transition.get_dim(name)
                                    if name in transition_sources
                                    else self.readout.get_dim(name)
                                    for name in self.readout.source_names]

        # Configure fork. For similar reasons as outlined above,
        # first push `readout` configuration.
        self.readout.push_allocation_config()
        feedback_name, = self.readout.feedback.outputs
        self.fork.input_dim = self.readout.get_dim(feedback_name)
        self.fork.output_dims = self.transition.get_dims(
            self.fork.apply.outputs)
        self.src_selector_fork.input_dim = self.get_dim('attended_1')
        self.src_selector_fork.output_dims = self.transition.get_dims(
            self.fork.apply.outputs)

    @application
    def cost_matrix(self, application_call, outputs, mask=None, **kwargs):
        """Returns generation costs for output sequences.

        See Also
        --------
        :meth:`cost` : Scalar cost.

        """
        # We assume the data has axes (time, batch, features, ...)
        batch_size = outputs.shape[1]

        # Prepare input for the iterative part
        states = dict_subset(kwargs, self._state_names, must_have=False)
        contexts = dict_subset(kwargs, self._context_names)
        feedback = self.readout.feedback(outputs)
        inputs = self.fork.apply(feedback, as_dict=True)

        # Prepare source selector input to gates
        input_src_sel, update_src_sel, reset_src_sel = \
            self.src_selector_fork.apply(kwargs['attended_1'])
        inputs['inputs'] += input_src_sel
        inputs['update_inputs'] += update_src_sel
        inputs['reset_inputs'] += reset_src_sel

        # Run the recurrent network
        results = self.transition.apply(
            mask=mask, return_initial_states=True, as_dict=True,
            **dict_union(inputs, states, contexts))

        # Separate the deliverables. The last states are discarded: they
        # are not used to predict any output symbol. The initial glimpses
        # are discarded because they are not used for prediction.
        # Remember, glimpses are computed _before_ output stage, states are
        # computed after.
        states = {name: results[name][:-1] for name in self._state_names}
        glimpses = {name: results[name][1:] for name in self._glimpse_names}

        # Compute the cost
        feedback = tensor.roll(feedback, 1, 0)
        feedback = tensor.set_subtensor(
            feedback[0],
            self.readout.feedback(self.readout.initial_outputs(batch_size)))
        readouts = self.readout.readout(
            feedback=feedback, **dict_union(states, glimpses, contexts))
        costs = self.readout.cost(readouts, outputs)
        if mask is not None:
            costs *= mask

        for name, variable in list(glimpses.items()) + list(states.items()):
            application_call.add_auxiliary_variable(
                variable.copy(), name=name)
        return costs
