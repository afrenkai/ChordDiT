from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from prototypes import DatasetSample, EditRecord
from transforms import CenterSquareCropTransform

DEFAULT_DATA_ROOT = Path(__file__).resolve().parent / "images"
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}


class LocalEditDataset(Dataset[DatasetSample]):
    def __init__(
        self,
        records: list[EditRecord],
        image_size: int = 512,
        use_center_crop: bool = False,
    ) -> None:
        if not records:
            raise ValueError("No records found in the dataset root")
        if image_size < 1:
            raise ValueError("image_size must be positive")

        self.records = records
        self.image_transform = transforms.Compose(
            [
                *([CenterSquareCropTransform()] if use_center_crop else []),
                transforms.Resize((image_size, image_size)),
            ]
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> DatasetSample:
        record = self.records[index]
        with Image.open(record.image_path) as source_image:
            image = self.image_transform(source_image.convert("RGB"))

        return DatasetSample(
            id=record.edit_id or record.image_path.stem,
            original_image=image,
            edited_image=Image.new("RGB", image.size, color="white"),
            original_prompt=record.source_prompt,
            edited_prompt=record.target_prompt,
            edit_prompt=record.edit_prompt,
            image_path=str(record.image_path),
        )


def load_local_dataset(
    path: str | Path | None = None,
    image_size: int = 512,
    center_crop: bool = True,
) -> LocalEditDataset:
    root = resolve_dataset_root(path)
    records = parse_edit_records(root)
    return LocalEditDataset(records, image_size, center_crop)


def resolve_dataset_root(path: str | Path | None) -> Path:
    root = Path(path).expanduser().resolve() if path is not None else DEFAULT_DATA_ROOT
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    return root


def parse_edit_records(root: Path) -> list[EditRecord]:
    records: list[EditRecord] = []
    for sample_directory in sorted(path for path in root.iterdir() if path.is_dir()):
        metadata_path = sample_directory / "meta.jsonl"
        if not metadata_path.exists():
            continue
        try:
            image_path = select_image_file(sample_directory)
        except FileNotFoundError:
            continue

        with metadata_path.open("r", encoding="utf-8") as metadata_file:
            for line_number, line in enumerate(metadata_file, start=1):
                if not line.strip():
                    continue
                try:
                    metadata = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid JSON in {metadata_path} at line {line_number}: {error}"
                    ) from error
                if not isinstance(metadata, dict):
                    raise TypeError(f"Expected an object in {metadata_path} at line {line_number}")

                target_prompt = str(metadata.get("edited_prompt", ""))
                records.append(
                    EditRecord(
                        image_path=image_path,
                        source_prompt=str(metadata.get("original_prompt", "")),
                        target_prompt=target_prompt,
                        edit_prompt=str(metadata.get("edit_prompt", target_prompt)),
                        edit_id=(
                            str(metadata["edit_id"])
                            if metadata.get("edit_id") is not None
                            else None
                        ),
                    )
                )

    if not records:
        raise FileNotFoundError(
            f"No edit samples found under {root}. Expected subdirectories with meta.jsonl files"
        )
    return records


def select_image_file(folder: Path) -> Path:
    candidates = sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not candidates:
        raise FileNotFoundError(f"No image found inside {folder}")

    preferred_names = {"i", "image", "original"}
    preferred = [path for path in candidates if path.stem.lower() in preferred_names]
    return preferred[0] if preferred else candidates[0]
