"""Dataset CRUD endpoints."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from ...core.models import Dataset, DatasetCreate
from ..deps import RepositoryDep

router = APIRouter(prefix="/datasets", tags=["datasets"])


@router.post("", response_model=Dataset, status_code=status.HTTP_201_CREATED)
async def create_dataset(body: DatasetCreate, repo: RepositoryDep) -> Dataset:
    """Register a new dataset of samples."""
    dataset = Dataset(name=body.name, samples=body.samples)
    return await repo.add_dataset(dataset)


@router.get("", response_model=list[Dataset])
async def list_datasets(repo: RepositoryDep) -> list[Dataset]:
    return await repo.list_datasets()


@router.get("/{dataset_id}", response_model=Dataset)
async def get_dataset(dataset_id: str, repo: RepositoryDep) -> Dataset:
    dataset = await repo.get_dataset(dataset_id)
    if dataset is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"dataset '{dataset_id}' not found",
        )
    return dataset
