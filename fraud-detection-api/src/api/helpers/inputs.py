"""fraud detection api input models."""

import pandas
import xgboost
from pydantic import BaseModel


class Input(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float

    def to_pandas(self) -> pandas.DataFrame:
        data = [
            [
                self.Time,
                self.V1,
                self.V2,
                self.V3,
                self.V4,
                self.V5,
                self.V6,
                self.V7,
                self.V8,
                self.V9,
                self.V10,
                self.V11,
                self.V12,
                self.V13,
                self.V14,
                self.V15,
                self.V16,
                self.V17,
                self.V18,
                self.V19,
                self.V20,
                self.V21,
                self.V22,
                self.V23,
                self.V24,
                self.V25,
                self.V26,
                self.V27,
                self.V28,
                self.Amount,
            ]
        ]

        columns = [
            "Time",
            "V1",
            "V2",
            "V3",
            "V4",
            "V5",
            "V6",
            "V7",
            "V8",
            "V9",
            "V10",
            "V11",
            "V12",
            "V13",
            "V14",
            "V15",
            "V16",
            "V17",
            "V18",
            "V19",
            "V20",
            "V21",
            "V22",
            "V23",
            "V24",
            "V25",
            "V26",
            "V27",
            "V28",
            "Amount",
        ]

        output = pandas.DataFrame(data=data, columns=columns)
        return output

    def to_dmatrix(self) -> xgboost.DMatrix:
        return xgboost.DMatrix(self.to_pandas())

    class Config:
        json_schema_extra = {
            "example": {
                "Time": 123549.0,
                "V1": 0.0324244087367978,
                "V2": 0.928688935050824,
                "V3": -0.343407283021625,
                "V4": -0.620324374697907,
                "V5": 0.977567841434462,
                "V6": -0.491908261338232,
                "V7": 0.912720484797719,
                "V8": 0.027674175919869,
                "V9": -0.386914146755369,
                "V10": -0.696476846256358,
                "V11": 1.24226394882851,
                "V12": 0.866450273409973,
                "V13": 0.344650141053576,
                "V14": -0.89265842134895,
                "V15": -0.947357528057424,
                "V16": 0.415335814534708,
                "V17": 0.190179644156345,
                "V18": 0.103573595171275,
                "V19": -0.0632374102529764,
                "V20": 0.0917928089460491,
                "V21": -0.258784981151548,
                "V22": -0.643562188406274,
                "V23": 0.142541234317062,
                "V24": 0.611408903485703,
                "V25": -0.459356058816613,
                "V26": 0.0903803875417543,
                "V27": 0.218078472698999,
                "V28": 0.077749433877637,
                "Amount": 17.99,
            }
        }
