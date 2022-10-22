
import json
import os


os.environ["TF_CONFIG"] = json.dumps(
    {
        "cluster":{
            "worker": ["host1:port", "host2:port", "host3:port"]
        },
        "task":{
            "type": "worker",
            "index": 1
        }
    }
)