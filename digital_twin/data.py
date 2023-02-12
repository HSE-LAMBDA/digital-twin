import pandera as pa
from pandera.typing import Series


""" Dependent variables used as target for the training process."""
POOL_DEPENDENT_VARS = ["iops", "lat"]
POOL_INDEPENDENT_VARS = ["block_size", "n_jobs", "iodepth", "read_fraction", "load_type", "io_type", "raid", "n_disks", "device_type", "offset"]

class PoolDataSchema(pa.SchemaModel):
    """ Data schema for the data.
    This schema is used to validate the data.
    """
    iops: Series[int] = pa.Field(gt=0, raise_warning=True)
    lat: Series[float] = pa.Field(gt=0, raise_warning=True)
    block_size: Series[int] = pa.Field(gt=0)
    n_jobs: Series[int] = pa.Field(gt=0)
    iodepth: Series[int] = pa.Field(gt=0)
    read_fraction: Series[int] = pa.Field(ge=0, le=100)
    load_type: Series[str] = pa.Field()
    io_type: Series[str] = pa.Field()
    raid: Series[str] = pa.Field()
    n_disks: Series[int] = pa.Field(gt=0)
    device_type: Series[str] = pa.Field()
    offset: Series[int] = pa.Field(ge=0)
    id: Series[str] = pa.Field()

    @pa.check("raid", "raid")
    def check_raid(cls, a: Series) -> bool:
        "RAID configuration must be in the following format 4+1"
        return a.str.contains(r"\d+\+\d+").all()
    
    @pa.check("load_type", "load_type")
    def check_load_type(cls, a: Series) -> bool:
        "Load type must be either load or unload"
        return a.map(lambda x: x in ('random', 'sequential')).all()
  
CacheDataSchema = PoolDataSchema
