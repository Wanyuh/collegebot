import json

import asyncpg
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
from typing import List, Optional, Dict
from starlette.responses import HTMLResponse
from starlette.staticfiles import StaticFiles

app = FastAPI()

# Database configuration
DB_USER = 'postgres'
DB_PASS = 'thisisatest'
DB_NAME = 'postgres'
DB_HOST = 'rag-project.cr482uc82ad7.us-west-2.rds.amazonaws.com'
DB_PORT = '5432'

app.mount("/static", StaticFiles(directory="static"), name="static")

# Model for Dataset entries
class Dataset(BaseModel):
    id: int
    name: str
    description: Optional[str] = None
    type: Optional[str] = None
    meta: Optional[Dict] = Field(default_factory=dict)  # Ensure this is a dict
    user_id: str
    remark: Optional[str] = None
    creator: Optional[str] = "default_creator"
    updater: Optional[str] = "default_updater"
    tenant_id: Optional[int]


async def get_db_connection():
    return await asyncpg.connect(
        user=DB_USER, password=DB_PASS, database=DB_NAME, host=DB_HOST, port=DB_PORT
    )

@app.get("/", response_class=HTMLResponse)
async def read_index():
    with open("static/max_kb_dataset_index.html") as f:
        return f.read()

# Create or update dataset entry
@app.put("/datasets/{dataset_id}")
async def create_or_update_dataset(dataset_id: int, dataset: Dataset):
    conn = await get_db_connection()
    try:
        query = """
            INSERT INTO max_kb_dataset (id, name, description, type, meta, user_id, remark, creator, updater, tenant_id, create_time, update_time)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            ON CONFLICT (id) DO UPDATE SET
                name = EXCLUDED.name,
                description = EXCLUDED.description,
                type = EXCLUDED.type,
                meta = EXCLUDED.meta,
                user_id = EXCLUDED.user_id,
                remark = EXCLUDED.remark,
                updater = EXCLUDED.updater,
                tenant_id = EXCLUDED.tenant_id,
                update_time = CURRENT_TIMESTAMP;
        """
        await conn.execute(
            query,
            dataset.id,
            dataset.name,
            dataset.description,
            dataset.type,
            dataset.meta,
            dataset.user_id,
            dataset.remark,
            dataset.creator,
            dataset.updater,
            dataset.tenant_id,
        )
        response = {
            "message": "Dataset entry created or updated successfully",
            "meta": dataset.meta  # Ensure meta is included as a dict
        }
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
    finally:
        await conn.close()


# Delete dataset entry
@app.delete("/datasets/{dataset_id}")
async def delete_dataset(dataset_id: int):
    conn = await get_db_connection()
    try:
        query = "DELETE FROM max_kb_dataset WHERE id = $1;"
        result = await conn.execute(query, dataset_id)
        if result == "DELETE 0":
            raise HTTPException(status_code=404, detail="Dataset entry not found")
        return {"message": "Dataset entry deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
    finally:
        await conn.close()


# Query single dataset entry by ID
@app.get("/datasets/{dataset_id}", response_model=Dataset)
async def get_dataset(dataset_id: int):
    conn = await get_db_connection()
    try:
        query = "SELECT * FROM max_kb_dataset WHERE id = $1;"
        result = await conn.fetchrow(query, dataset_id)
        print(result)
        if result is None:
            raise HTTPException(status_code=404, detail="Dataset entry not found")

        dataset_dict = dict(result)
        # Parse the meta field from a JSON string to a dictionary
        if isinstance(dataset_dict['meta'], str):
            try:
                dataset_dict['meta'] = json.loads(dataset_dict['meta'])
            except json.JSONDecodeError:
                dataset_dict['meta'] = {}  # Handle case where JSON is invalid

        # Ensure meta is a dict; if it's None or not a dict, replace it with an empty dict
        if not isinstance(dataset_dict['meta'], dict):
            dataset_dict['meta'] = {}

        return dataset_dict
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
    finally:
        await conn.close()


# Query all dataset entries
@app.get("/datasets", response_model=List[Dataset])
async def get_all_datasets():
    conn = await get_db_connection()
    try:
        query = "SELECT * FROM max_kb_dataset;"
        result = await conn.fetch(query)
        datasets = []
        for row in result:
            dataset_dict = dict(row)

            # Parse the meta field from a JSON string to a dictionary
            if isinstance(dataset_dict['meta'], str):
                try:
                    dataset_dict['meta'] = json.loads(dataset_dict['meta'])
                except json.JSONDecodeError:
                    dataset_dict['meta'] = {}  # Handle invalid JSON

            # Ensure meta is a dict; if it's None or not a dict, replace it with an empty dict
            if not isinstance(dataset_dict['meta'], dict):
                dataset_dict['meta'] = {}

            datasets.append(dataset_dict)  # Append to the datasets list

        return datasets
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
    finally:
        await conn.close()

if __name__ == "__main__":

    uvicorn.run(app, host="0.0.0.0", port=8001)
