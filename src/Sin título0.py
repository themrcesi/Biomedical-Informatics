# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 17:13:27 2020

@author: César
"""


import os
import pandas as pd
import dask.bag as db
import json

def flatten(doc):
    return {
        "id": doc["paper_id"],
        "abstract": " ".join(doc["abtract"]["text"]),
        "text": " ".join(doc["body_text"]["text"])
    }

docs = db.read_text("B:/document_parser/document_parses/pdf_json/*.json").map(json.loads)

b = docs.map(flatten).to_dataframe()
