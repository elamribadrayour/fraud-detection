"""prepare data for fraud detection."""

import os
import logging
from zipfile import ZipFile

import pandas
import requests
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

import helpers.paths
import helpers.logger


def get_url() -> str:
    return "https://storage.googleapis.com/kaggle-data-sets/310/23498/bundle/archive.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20240928%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20240928T081017Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=a7f7425f2ea4202a48758da0a80acd673850d708888782e343cc65780e01a970545269c13fdf3ed28e013ad03f25920a20379c156ed3d3186c4f09dc096e70f09214b0bdca819115736fc756c9a1fa1704e43947a27fafc53c0f666e17d49b6912454c7de9760da4ed084a65bfb29bf068cf6a55ac284a79b8660bfd1c5260a08c6dcd46ac19d4d5ff85ab442517eefc2d6dffe224ded1e6b927917b5a5d9a06e2dc570dfe6442c8b55898cffbe1c865be53f00bb8ae38bd4611c027c25d6f3eeaeb5a3086fe3404c9761306c26d77fa6720c894f3f97947abbf804eefbaa3b2d42e752ecc926d9040e92b3f79b7f805289bd36a515c9b74303fb68796a04bf8"


def get_headers() -> dict[str, str]:
    return {
        "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
        "accept-language": "en-GB,en-US;q=0.9,en;q=0.8,fr;q=0.7",
        "dnt": "1",
        "priority": "u=0, i",
        "referer": "https://www.kaggle.com/",
        "sec-ch-ua": '"Google Chrome";v="129", "Not=A?Brand";v="8", "Chromium";v="129"',
        "sec-ch-ua-arch": '"arm"',
        "sec-ch-ua-bitness": '"64"',
        "sec-ch-ua-full-version-list": '"Google Chrome";v="129.0.6668.59", "Not=A?Brand";v="8.0.0.0", "Chromium";v="129.0.6668.59"',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-model": '""',
        "sec-ch-ua-platform": '"macOS"',
        "sec-ch-ua-platform-version": '"14.6.1"',
        "sec-ch-ua-wow64": "?0",
        "sec-fetch-dest": "document",
        "sec-fetch-mode": "navigate",
        "sec-fetch-site": "cross-site",
        "sec-fetch-user": "?1",
        "upgrade-insecure-requests": "1",
        "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36",
        "x-browser-channel": "stable",
        "x-browser-copyright": "Copyright 2024 Google LLC. All rights reserved.",
        "x-browser-validation": "IrKrQbZgwzfhwxeKC1pprn3FhX8=",
        "x-browser-year": "2024",
        "x-client-data": "CJe2yQEIprbJAQipncoBCIGJywEIkqHLAQia/swBCIagzQEI3L7OARjBy8wB",
    }


def download_data(cache_path: str) -> str:
    local_path = os.path.join(
        helpers.paths.get_data_path(cache_path=cache_path), "raw.zip"
    )
    if os.path.exists(local_path) is True:
        logging.info(f"loading local file {local_path}")
        return local_path

    response = requests.get(
        url=get_url(),
        headers=get_headers(),
    )
    with open(local_path, mode="wb") as f:
        f.write(response.content)

    logging.info(f"file saved to {local_path}")
    return local_path


def extract_data(zip_path: str, cache_path: str) -> str:
    local_path = helpers.paths.get_data_path(cache_path=cache_path)
    if os.path.exists(zip_path) is False:
        raise IOError(f"file {zip_path} doesn't exist")

    with ZipFile(zip_path, "r") as ref:
        ref.extractall(local_path)

    old_path = os.path.join(local_path, "creditcard.csv")
    new_path = os.path.join(local_path, "raw.csv")
    os.rename(old_path, new_path)
    return new_path


def get_raw_data(cache_path: str) -> pandas.DataFrame:
    local_path = os.path.join(
        helpers.paths.get_data_path(cache_path=cache_path), "raw.parquet"
    )
    if os.path.exists(local_path) is True:
        return pandas.read_parquet(path=local_path)

    zip_path = download_data(cache_path=cache_path)
    csv_path = extract_data(zip_path=zip_path, cache_path=cache_path)
    output = pandas.read_csv(filepath_or_buffer=csv_path)
    output.to_parquet(path=local_path, compression="gzip", engine="pyarrow")
    return output


def set_train_test_data(data: pandas.DataFrame, cache_path: str) -> None:
    train_path = os.path.join(
        helpers.paths.get_data_path(cache_path=cache_path),
        "train.parquet",
    )
    test_path = os.path.join(
        helpers.paths.get_data_path(cache_path=cache_path),
        "test.parquet",
    )

    df_train, df_test = train_test_split(data)

    x_train = df_train.drop("Class", axis=1)
    y_train = df_train["Class"]

    smote = SMOTE(random_state=42)
    x_train_sm, y_train_sm = smote.fit_resample(x_train, y_train)
    df_train = pandas.concat(objs=[x_train_sm, y_train_sm], axis=1)
    logging.info("applied SMOTE for balancing the dataset")

    df_test.to_parquet(path=test_path, compression="gzip", engine="pyarrow")
    df_train.to_parquet(path=train_path, compression="gzip", engine="pyarrow")
    return


def run(cache_path: str) -> None:
    helpers.logger.init()
    logging.info("running data preparation for fraud detection")
    data = get_raw_data(cache_path=cache_path)
    set_train_test_data(data=data, cache_path=cache_path)
