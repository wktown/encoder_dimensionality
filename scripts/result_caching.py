from collections import defaultdict, OrderedDict

import inspect
import itertools
import logging
import numpy as np
import pandas as pd
import os
import pickle
import xarray as xr
from functools import wraps
from typing import Union


def get_function_identifier(function, call_args):
    module = [function.__module__, function.__name__]
    if 'self' in call_args:
        object = call_args['self']
        class_name = object.__class__.__name__
        if 'object at' in str(object):
            object = class_name
        else:
            object = f"{class_name}({str(object)})"
        module.insert(1, object)
        del call_args['self']
    module = '.'.join(module)
    strip_slashes = lambda x: str(x).replace('/', '_')
    params = ','.join(f'{key}={strip_slashes(value)}' for key, value in call_args.items())
    if params:
        function_identifier = os.path.join(module, params)
    else:
        function_identifier = module
    return function_identifier


def is_enabled(function_identifier):
    disable = os.getenv('RESULTCACHING_DISABLE', '0')
    return not _match_identifier(function_identifier, disable)


def cached_only(function_identifier):
    cachedonly = os.getenv('RESULTCACHING_CACHEDONLY', '0')
    return _match_identifier(function_identifier, cachedonly)


def _match_identifier(function_identifier, match_value):
    if match_value == '1':
        return True
    if match_value == '':
        return False
    disabled_modules = match_value.split(',')
    return any(function_identifier.startswith(disabled_module) for disabled_module in disabled_modules)


class NotCachedError(Exception):
    pass

class _DiskStorage(_Storage):
    def __init__(self, identifier_ignore=()):
        super().__init__(identifier_ignore=identifier_ignore)
        self._storage_directory = os.path.expanduser(os.getenv('RESULTCACHING_HOME', '~/.result_caching'))

    def storage_path(self, function_identifier):
        return os.path.join(self._storage_directory, function_identifier + '.pkl')

    def save(self, result, function_identifier):
        path = self.storage_path(function_identifier)
        path_dir = os.path.dirname(path)
        if not os.path.isdir(path_dir):
            os.makedirs(path_dir, exist_ok=True)
        savepath_part = path + '.filepart'
        self.save_file(result, savepath_part)
        os.rename(savepath_part, path)

    def save_file(self, result, savepath_part):
        with open(savepath_part, 'wb') as f:
            pickle.dump({'data': result}, f, protocol=-1)  # highest protocol

    def is_stored(self, function_identifier):
        storage_path = self.storage_path(function_identifier)
        return os.path.isfile(storage_path)

    def load(self, function_identifier):
        path = self.storage_path(function_identifier)
        assert os.path.isfile(path)
        return self.load_file(path)

    def load_file(self, path):
        with open(path, 'rb') as f:
            return pd.read_pickle(f)['data']
        

class _DictStorage(_DiskStorage):
    """
    All fields in _combine_fields are combined into one file and loaded lazily
    """

    def __init__(self, dict_key: str, *args, **kwargs):
        """
        :param dict_key: the argument representing the dictionary key.
        """
        super().__init__(*args, **kwargs)
        self._dict_key = dict_key

    def __call__(self, function):
        def wrapper(*args, **kwargs):
            call_args = self.getcallargs(function, *args, **kwargs)
            assert self._dict_key in call_args
            infile_call_args = {self._dict_key: call_args[self._dict_key]}
            function_identifier = self.get_function_identifier(function, call_args)
            stored_result, reduced_call_args = None, call_args
            if is_enabled(function_identifier) and self.is_stored(function_identifier):
                self._logger.debug(f"Loading from storage: {function_identifier}")
                stored_result = self.load(function_identifier)
                infile_missing_call_args = self.missing_call_args(infile_call_args, stored_result)
                if len(infile_missing_call_args) == 0:
                    # nothing else to run, but still need to filter
                    result = stored_result
                    reduced_call_args = None
                else:
                    # need to run more args
                    non_variable_call_args = {key: value for key, value in call_args.items() if key != self._dict_key}
                    infile_missing_call_args = {self._dict_key: infile_missing_call_args}
                    reduced_call_args = {**non_variable_call_args, **infile_missing_call_args}
                    self._logger.debug(f"Computing missing: {reduced_call_args}")
            if reduced_call_args:
                if cached_only(function_identifier):
                    raise NotCachedError(f"The following arguments for '{function_identifier}' "
                                         f"are not stored: {reduced_call_args}")
                # run function if some args are uncomputed
                self._logger.debug(f"Running function: {function_identifier}")
                result = function(**reduced_call_args)
                if not self.callargs_present(result, {self._dict_key: reduced_call_args[self._dict_key]}):
                    raise ValueError("result does not contain requested keys")
                if stored_result is not None:
                    result = self.merge_results(stored_result, result)
                # only save if new results
                if is_enabled(function_identifier):
                    self._logger.debug("Saving to storage: {}".format(function_identifier))
                    self.save(result, function_identifier)
            assert self.callargs_present(result, infile_call_args)
            result = self.filter_callargs(result, infile_call_args)
            return result

        return wrapper

    def merge_results(self, stored_result, result):
        return {**stored_result, **result}

    def callargs_present(self, result, infile_call_args):
        # make sure coords are set equal to call_args
        return len(self.missing_call_args(infile_call_args, result)) == 0

    def missing_call_args(self, call_args, data):
        assert len(call_args) == 1 and list(call_args.keys())[0] == self._dict_key
        keys = list(call_args.values())[0]
        return [key for key in keys if key not in data]

    def filter_callargs(self, data, call_args):
        assert len(call_args) == 1 and list(call_args.keys())[0] == self._dict_key
        keys = list(call_args.values())[0]
        return type(data)((key, value) for key, value in data.items() if key in keys)
    
store = _DiskStorage
store_dict = _DictStorage