# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Implementation of parent level data generation pipeline with data-io."""

import os
import bz2
import dill
import zstd


class DataPipeline:
    """Main Pipeline Class"""

    def __init__(self) -> None:
        self.filepath = None

    @staticmethod
    def read_data(filepath):
        """Reading the data from a saved file"""
        with open(filepath, "rb") as file:
            compressed_pickle = file.read()
        depressed_pickle = bz2.decompress(compressed_pickle)
        data = dill.loads(depressed_pickle)
        return data

    @staticmethod
    def write_data(data, filepath, protocol=4):
        """Writing the data to a file"""
        pickled_data = dill.dumps(data, protocol=protocol)  # returns data as a bytes object
        compressed_pickle = zstd.compress(pickled_data)
        if not os.path.exists(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath))
        with open(filepath + "_full.dat", "wb") as file:
            file.write(compressed_pickle)

    @staticmethod
    def write_data_seperated(data, filepath, protocol=4):
        """Writing the data to a file"""
        # print(data)
        for key, val in data.items():
            pickled_data = dill.dumps(val, protocol=protocol)  # returns data as a bytes object
            compressed_pickle = zstd.compress(pickled_data)
            with open(filepath + f"_{key}.dat", "wb") as file:
                file.write(compressed_pickle)

    @staticmethod
    def append_data(filepath):
        """Append the data to a file"""
        try:
            with open(filepath, "rb") as file:
                compressed_pickle = file.read()
            depressed_pickle = bz2.decompress(compressed_pickle)
            data = dill.loads(depressed_pickle)
        except OSError:
            data = {}
        return data
