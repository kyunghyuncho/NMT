
import logging
import numpy
import os
import re
import time

from contextlib import closing
from toolz import merge

from blocks.algorithms import DifferentiableCostMinimizer
from blocks.extensions import SimpleExtension, TrainingExtension
from blocks.monitoring.evaluators import AggregationBuffer
from blocks.extensions.monitoring import MonitoringExtension
from blocks.serialization import secure_dump, load, BRICK_DELIMITER
from blocks.utils import reraise_as

logger = logging.getLogger(__name__)

LOADED_FROM = "loaded_from"
SAVED_TO = "saved_to"


class PrintMultiStream(SimpleExtension):
    """Prints number of batches seen for each data stream"""
    def __init__(self, **kwargs):
        super(PrintMultiStream, self).__init__(**kwargs)

    def do(self, which_callback, *args):
        counters = self.main_loop.data_stream.training_counter
        epochs = self.main_loop.data_stream.epoch_counter
        sid = self.main_loop.data_stream.curr_id
        src_size = args[0]['source'].shape
        trg_size = args[0]['target'].shape
        msg = ['Source_{}:iter[{}]-epoch[{}]'.format(i, c, e)
               for i, (c, e) in enumerate(zip(counters, epochs))]
        print("Multi-stream status:")
        print "\t", "Using stream: source_{}".format(sid)
        print "\t", "Source shape: {}".format(src_size)
        print "\t", "Target shape: {}".format(trg_size)
        print "\t", " ".join(msg)


class IncrementalDump(SimpleExtension):
    """Incrementally dumps model given frequency."""

    def __init__(self, saveto, **kwargs):
        raise NotImplementedError("To be implemented!")
        super(IncrementalDump, self).__init__(**kwargs)
        self.saveto = saveto
        self.modelID = self._get_model_id(saveto)

    def _get_model_id(self, saveto):
        try:
            postfix = [int(m.group(1))
                       for m in [re.match(r'.*_([-0-9]+)', f)
                                 for f in os.listdir(saveto)]
                       if m is not None]
            model_id = max(postfix)
        except:
            model_id = 0
        return model_id

    def do(self, which_callback, *args):
        pass


class TrainingDataMonitoringWithMultiCG(SimpleExtension, MonitoringExtension):
    """Monitors and logs variables for multi CG."""

    def __init__(self, variables, **kwargs):
        """Variables should be a list of list."""
        num_cgs = len(variables)
        kwargs.setdefault("before_training", True)
        super(TrainingDataMonitoringWithMultiCG, self).__init__(**kwargs)
        self._buffers = []
        for i in xrange(num_cgs):
            self._buffers.append(
                AggregationBuffer(variables[i]
                                  if isinstance(variables[i], list)
                                  else [variables[i]],
                                  use_take_last=True))
        self._last_time_called = -1

    def do(self, callback_name, *args):
        if callback_name == 'before_training':
            for i in xrange(self.main_loop.num_cgs):
                if not isinstance(self.main_loop.algorithm.algorithms[i],
                                  DifferentiableCostMinimizer):
                    raise ValueError
                self.main_loop.algorithm.algorithms[i].add_updates(
                    self._buffers[i].accumulation_updates)
                self._buffers[i].initialize_aggregators()
        else:
            if (self.main_loop.status['iterations_done'] ==
                    self._last_time_called):
                raise Exception("TrainingDataMonitoring.do should be invoked"
                                " no more than once per iteration")
            self._last_time_called = self.main_loop.status['iterations_done']
            enc_id = numpy.argmax(args[0]['src_selector'])
            records = self._buffers[enc_id].get_aggregated_values().items()

            recs_ = []
            for rec in records:
                if rec[-1].shape > 0:
                    recs_.append((rec[0], rec[-1].mean()))
                else:
                    recs_.append(rec)
            if any([numpy.isnan(rec[-1]) or numpy.isinf(rec[-1])
                    for rec in recs_]):
                import ipdb;ipdb.set_trace()
                logger.error('!!!Non-finite element!!!')
            self.add_records(self.main_loop.log, records)
            self._buffers[enc_id].initialize_aggregators()


class SimpleTrainingDataMonitoringWithMultiCG(SimpleExtension,
                                              MonitoringExtension):
    """Alternative monitor for TrainingDataMonitoring."""

    def __init__(self, **kwargs):
        super(SimpleTrainingDataMonitoringWithMultiCG, self).__init__(**kwargs)
        self._last_time_called = -1

    def do(self, callback_name, *args):
        if (self.main_loop.status['iterations_done'] ==
                self._last_time_called):
            raise Exception("TrainingDataMonitoring.do should be invoked"
                            " no more than once per iteration")
        self._last_time_called = self.main_loop.status['iterations_done']
        enc_id = numpy.argmax(args[0]['src_selector'])
        self.add_records(
            self.main_loop.log,
            self.main_loop.algorithm.retvals[enc_id].items())


class MainLoopDumpManagerWMT15(object):
    """Checkpointintg for multi CG main loop."""

    def __init__(self, saveto, save_accumulators=False,
                 load_accumulators=False):
        self.folder = saveto
        self.save_accumulators = save_accumulators
        self.load_accumulators = load_accumulators

    @property
    def path_to_accumulators(self):
        return os.path.join(self.folder, 'algo{}.npz')

    @property
    def path_to_parameters(self):
        return os.path.join(self.folder, 'params.npz')

    @property
    def path_to_iteration_state(self):
        return os.path.join(self.folder, 'iterations_state.pkl')

    @property
    def path_to_log(self):
        # The extension is omitted for the log because advanced
        # log classes might have a better format for storing on the disk
        # then pickled file. Or alternatively, log will be dump as pure
        # text file of (time, key, value) triples. Currenly log is just
        # pickled though.
        return os.path.join(self.folder, 'log')

    def dump_iteration_state(self, main_loop):
        secure_dump(main_loop.iteration_state, self.path_to_iteration_state)

    def dump_log(self, main_loop):
        secure_dump(main_loop.log, self.path_to_log)

    def load_parameters(self):
        return load_parameter_values(self.path_to_parameters)

    def load_iteration_state(self):
        with open(self.path_to_iteration_state, "rb") as source:
            return load(source)

    def load_log(self):
        with open(self.path_to_log, "rb") as source:
            return load(source)

    def load(self):
        return (self.load_parameters(),
                self.load_iteration_state(),
                self.load_log())

    def load_to(self, main_loop):
        """Loads the dump from the root folder into the main loop."""
        logger.info(" Reloading model")
        try:
            logger.info(" ...loading model parameters")
            params_all = self.load_parameters()
            for i in xrange(main_loop.num_cgs):
                params_this = main_loop.models[i].get_parameter_dict()
                missing = set(params_this.keys()) - set(params_all.keys())
                for pname in params_this.keys():
                    if pname in params_all:
                        val = params_all[pname]
                        if params_this[pname].get_value().shape != val.shape:
                            logger.warning(
                                " Dimension mismatch {}-{} for {}"
                                .format(params_this[pname].get_value().shape,
                                        val.shape, pname))

                        params_this[pname].set_value(val)
                        logger.info(" Loaded to CG[{}] {:15}: {}"
                                    .format(i, val.shape, pname))
                    else:
                        logger.warning(
                            " Parameter does not exist: {}".format(pname))

                logger.info(
                    " Number of parameters loaded for computation graph[{}]: {}"
                    .format(i, len(params_this) - len(missing)))
        except Exception as e:
            logger.error(" Error {0}".format(str(e)))

        try:
            logger.info(" Loading iteration state...")
            main_loop.iteration_state = self.load_iteration_state()
        except Exception as e:
            logger.error(" Error {0}".format(str(e)))

        try:
            logger.info(" Loading log...")
            main_loop.log = self.load_log()
        except Exception as e:
            logger.error(" Error {0}".format(str(e)))

        if self.load_accumulators:
            try:
                logger.info(" Loading algorithm accumulators...")
                self._load_accumulators(main_loop)
            except Exception as e:
                logger.error(" Error {0}".format(str(e)))

    def dump_parameters(self, main_loop):
        params_to_save = []
        for i in xrange(main_loop.num_cgs):
            params_to_save.append(
                main_loop.models[i].get_parameter_values())
        save_parameter_values(merge(params_to_save),
                              self.path_to_parameters)

    def dump_accumulators(self, main_loop):
        """Each step rule has different number of accumulators"""
        for i in xrange(main_loop.num_cgs):
            algo = main_loop.algorithm.algorithms[i]
            accums = algo.step_rule_updates
            params = algo.steps.items()
            model_params = main_loop.models[i].get_parameter_dict()

            # Reshape this long list into (num_params, num_accums_per_param)
            num_params = len(params)
            num_accums = len(accums)
            assert num_accums % num_params == 0, \
                "Accumulators cannot be loaded for CG[{}]".format(i)

            # This is num_accums_per_param
            col = num_accums / num_params
            accums_mat = [accums[col*l:col*(l+1)] for l in range(num_params)]
            accums_vals = [[y[0].get_value() for y in x] for x in accums_mat]

            # Get corresponding parameter names and create a dictionary
            names = [[k for k, v in model_params.iteritems()
                      if v == params[l][0]][0] for l in xrange(len(params))]
            params_dict = dict([(names[l].replace("/", "-"), accums_vals[l])
                                for l in xrange(len(names))])

            # Save here
            numpy.savez(self.path_to_accumulators.format(i), **params_dict)

    def dump(self, main_loop):
        """Overwrites MainLoopDumpManager.dump()."""
        if not os.path.exists(self.folder):
            os.mkdir(self.folder)
        print ""
        logger.info(" Saving model")
        start = time.time()
        logger.info(" ...saving parameters")
        self.dump_parameters(main_loop)
        logger.info(" ...saving iteration state")
        self.dump_iteration_state(main_loop)
        logger.info(" ...saving log")
        self.dump_log(main_loop)
        if self.save_accumulators:
            logger.info(" ...saving algorithm")
            self.dump_accumulators(main_loop)
        logger.info(" Model saved, took {} seconds.".format(time.time()-start))

    def _load_accumulators(self, main_loop):
        """Nasty method, use carefully"""
        for i in xrange(main_loop.num_cgs):
            source = numpy.load(self.path_to_accumulators.format(i))
            accums_dict = {name.replace("-", "/"): value
                           for name, value in source.items()}
            source.close()
            algo = main_loop.algorithm.algorithms[i]
            model_params = main_loop.models[i].get_parameter_dict()
            steps = algo.steps.items()

            for pidx in xrange(len(steps)):
                # Get parameter name and its accumulators
                p = steps[pidx][0]
                name = [k for k, v in model_params.iteritems() if v == p][0]
                accums = accums_dict[name]

                # This is num_accums_per_param
                col = len(accums)
                for aidx in xrange(col):
                    algo.step_rule_updates[pidx*col+aidx][0].set_value(
                        accums[aidx])


class DumpWithMultiCG(SimpleExtension):
    """Dumps the state of the main loop.
    Makes a `SAVED_TO` record in the log with the dumping destination
    in the case of success and ``None`` in the case of failure.
    Parameters
    ----------
    state_path : str
        The folder to dump the state to. Will be created it does not
        exist.
    Notes
    -----
    This is an extension of old saveload.Dump.
    Requires the model to be a Brick or a list of Bricks.
    """
    def __init__(self, saveto, save_accumulators=False, **kwargs):
        kwargs.setdefault("after_training", True)
        super(DumpWithMultiCG, self).__init__(**kwargs)
        self.manager = MainLoopDumpManagerWMT15(
            saveto, save_accumulators=save_accumulators)

    def do(self, callback_name, *args, **kwargs):
        try:
            self.main_loop.log.current_row[SAVED_TO] = (
                self.manager.folder)
            self.manager.dump(self.main_loop)
        except Exception:
            self.main_loop.log.current_row[SAVED_TO] = None
            raise


class LoadFromDumpMultiCG(TrainingExtension):
    """Wrapper to use MainLoopDumpManagerWMT15"""

    def __init__(self, saveto, load_accumulators=False, **kwargs):
        super(LoadFromDumpMultiCG, self).__init__(saveto, **kwargs)
        self.manager = MainLoopDumpManagerWMT15(
            saveto, load_accumulators=load_accumulators)

    def before_training(self):
        if not os.path.exists(self.manager.folder):
            logger.info("No dump found")
            return
        logger.info("Loading the state from {} into the main loop"
                    .format(self.manager.folder))
        try:
            self.manager.load_to(self.main_loop)
            self.main_loop.log.current_row[LOADED_FROM] = self.manager.folder
        except Exception:
            reraise_as("Failed to load the state")


def load_parameter_values(path):
    """Load parameter values saved by :func:`dump`.

    This is a thin wrapper over :func:`numpy.load`. It changes the names of
    the arrays to ones compatible with :meth:`.Model.set_param_values`.

    Parameters
    ----------
    path : str or file
        The source for loading from.

    Returns
    -------
    A dictionary of (parameter name, numpy array) pairs.

    """
    with closing(numpy.load(path)) as source:
        param_values = {}
        for name, value in source.items():
            if name != 'pkl':
                name_ = name.replace(BRICK_DELIMITER, '/')
                if not name_.startswith('/'):
                    name_ = '/' + name_
                param_values[name_] = value
    return param_values


def save_parameter_values(param_values, path):
    """Compactly save parameter values.
    This is a thin wrapper over `numpy.savez`. It deals with
    `numpy`'s vulnerability to slashes in file names.
    Parameters
    ----------
    param_values : dict of (parameter name, numpy array)
        The parameter values.
    path : str of file
        The destination for saving.
    """
    param_values = {name.replace("/", "-"): param
                    for name, param in param_values.items()}
    numpy.savez(path, **param_values)