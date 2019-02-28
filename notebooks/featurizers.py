# Databricks notebook source

from collections import defaultdict
from itertools import combinations, product
from mmh3 import hash as _mh3
from pyspark import keyword_only
from pyspark.ml import Transformer
from pyspark.ml.linalg import SparseVector, VectorUDT
from pyspark.ml.param.shared import (
    HasInputCol,
    HasInputCols,
    HasNumFeatures,
    HasOutputCol,
    HasSeed,
)
from pyspark.ml.pipeline import Pipeline
from pyspark.sql import functions as F, types as T

# COMMAND ----------

class ArrayHasher(Transformer, HasSeed, HasInputCol, HasOutputCol):

    @keyword_only
    def __init__(self, seed=0, inputCol=None, outputCol=None):
        super(ArrayHasher, self).__init__()
        self._setDefault(seed=0)
        self.setParams(**self._input_kwargs)

    @keyword_only
    def setParams(self, seed=0, inputCol=None, outputCol=None):
        return self._set(**self._input_kwargs)

    def _transform(self, dataset):
        inputCol = self.getInputCol()
        dataType = dataset.schema[inputCol].dataType
        assert isinstance(dataType, T.ArrayType)
        assert isinstance(dataType.elementType, T.StringType)
        seed = _mh3(inputCol, seed=self.getSeed())

        @F.udf(T.MapType(T.IntegerType(), T.FloatType()))
        def hash_(v):
            if not v:
                return {}
            hashVector = defaultdict(float)
            for x in v:
                h = _mh3(x, seed=seed)
                hashVector[h] += 1.
            return dict(hashVector)

        return dataset.withColumn(self.getOutputCol(),
            hash_(dataset[inputCol]))

# COMMAND ----------

class MapHasher(Transformer, HasSeed, HasInputCol, HasOutputCol):

    @keyword_only
    def __init__(self, seed=0, inputCol=None, outputCol=None):
        super(MapHasher, self).__init__()
        self._setDefault(seed=0)
        self.setParams(**self._input_kwargs)

    @keyword_only
    def setParams(self, seed=0, inputCol=None, outputCol=None):
        return self._set(**self._input_kwargs)

    def _transform(self, dataset):
        inputCol = self.getInputCol()
        dataType = dataset.schema[inputCol].dataType
        assert isinstance(dataType, T.MapType)
        assert isinstance(dataType.keyType, T.StringType)
        assert isinstance(dataType.valueType, (T.NumericType, T.StringType))
        seed = _mh3(inputCol, seed=self.getSeed())

        @F.udf(T.MapType(T.IntegerType(), T.FloatType()))
        def hashNumeric(v):
            if not v:
                return {}
            hashVector = defaultdict(float)
            for k, v in v.items():
                h = _mh3(k, seed=seed)
                hashVector[h] += v
            return dict(hashVector)

        @F.udf(T.MapType(T.IntegerType(), T.FloatType()))
        def hashString(v):
            if not v:
                return v
            hashVector = defaultdict(float)
            for k, v in v.items():
                h = _mh3(v, seed=_mh3(k, seed=seed))
                hashVector[h] += 1.
            return dict(hashVector)

        if isinstance(dataType.valueType, T.NumericType):
            return dataset.withColumn(self.getOutputCol(),
                hashNumeric(dataset[inputCol]))
        else:
            return dataset.withColumn(self.getOutputCol(),
                hashString(dataset[inputCol]))

# COMMAND ----------

class ScalarHasher(Transformer, HasSeed, HasInputCol, HasOutputCol):

    @keyword_only
    def __init__(self, seed=0, inputCol=None, outputCol=None):
        super(ScalarHasher, self).__init__()
        self._setDefault(seed=0)
        self.setParams(**self._input_kwargs)

    @keyword_only
    def setParams(self, seed=0, inputCol=None, outputCol=None):
        return self._set(**self._input_kwargs)

    def _transform(self, dataset):
        inputCol = self.getInputCol()
        dataType = dataset.schema[inputCol].dataType
        assert isinstance(dataType,
                          (T.BooleanType, T.NumericType, T.StringType))
        seed = _mh3(inputCol, seed=self.getSeed())

        @F.udf(T.MapType(T.IntegerType(), T.FloatType()))
        def hashNumeric(v):
            if not v:
                return {}
            return {seed: float(v)}

        @F.udf(T.MapType(T.IntegerType(), T.FloatType()))
        def hashString(v):
            if not v:
                return {}
            return {_mh3(v, seed=seed): 1.}

        if isinstance(dataType, (T.BooleanType, T.NumericType)):
            return dataset.withColumn(self.getOutputCol(),
                hashNumeric(dataset[inputCol]))
        else:
            return dataset.withColumn(self.getOutputCol(),
                hashString(dataset[inputCol]))

# COMMAND ----------

class PairwiseHashCrosser(Transformer, HasInputCols, HasOutputCol):

    @keyword_only
    def __init__(self, inputCols=None, outputCol=None):
        super(PairwiseHashCrosser, self).__init__()
        self.setParams(**self._input_kwargs)

    @keyword_only
    def setParams(self, inputCols=None, outputCol=None):
        return self._set(**self._input_kwargs)

    def _transform(self, dataset):

        @F.udf(T.MapType(T.IntegerType(), T.FloatType()))
        def cross(*values):
            if not values:
                return {}
            hashVector = defaultdict(float)
            for d1, d2 in combinations(values, 2):
                if not d1 or not d2:
                    continue
                for (k1, v1), (k2, v2) in product(d1.items(), d2.items()):
                    h = (k1 ^ k2)
                    hashVector[h] += v1 * v2
            return dict(hashVector)

        return dataset.withColumn(self.getOutputCol(),
            cross(*dataset[self.getInputCols()]))

# COMMAND ----------

class HashProjector(Transformer, HasNumFeatures, HasInputCols, HasOutputCol):

    @keyword_only
    def __init__(self, numFeatures=1 << 18, inputCols=None, outputCol=None):
        super(HashProjector, self).__init__()
        self._setDefault(numFeatures=1 << 18)
        self.setParams(**self._input_kwargs)

    @keyword_only
    def setParams(self, numFeatures=1 << 18, inputCols=None, outputCol=None):
        return self._set(**self._input_kwargs)

    def _transform(self, dataset):
        size = self.getNumFeatures()

        @udf(VectorUDT())
        def project(*values):
            if not values:
                return SparseVector(size, {})
            hashVector = defaultdict(float)
            for d in values:
                if not d:
                    continue
                for h, v in d.items():
                    if not v:
                        continue
                    k = abs(h) & (size - 1)
                    s = (h >= 0) * 2 - 1
                    hashVector[k] += v * s
            return SparseVector(size, dict(hashVector))

        return dataset.withColumn(self.getOutputCol(),
            project(*dataset[self.getInputCols()]))

# COMMAND ----------

featurizationPipeline = Pipeline(stages=[
    ScalarHasher(inputCol='movieId', outputCol='movieIdHashed'),
    ScalarHasher(inputCol='releaseYear', outputCol='releaseYearHashed'),
    ArrayHasher(inputCol='genres', outputCol='genresHashed'),
    MapHasher(inputCol='genreTFIDFs', outputCol='genreTFIDFsHashed'),
    PairwiseHashCrosser(inputCols=['genresHashed', 'genreTFIDFsHashed'],
                        outputCol='genreCross'),
    HashProjector(inputCols=['movieIdHashed', 'releaseYearHashed',
                             'genresHashed', 'genreTFIDFsHashed',
                             'genreCross'],
                  outputCol='features'),
])
