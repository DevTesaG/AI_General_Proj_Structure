SCHEMA = {
  "type": "object",
  "properties": {
    "image":{
        "type":"array",
        "items":{
            "type": "array",
            "items": {
                "type": "array",
                "items": {
                    "type": "array",
                    "items": {
                        "type": "number"
                    }
                }
            }
        }
    }
  },
  "required":["image"]
}