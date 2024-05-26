FROM python:3.10

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all the content of the current directory to the working directory in the container
COPY . .

# Expose port 8080 to access the application
EXPOSE 8080

# Command to run the FastAPI application
CMD ["uvicorn", "challenge_MLE.challenge.api:app", "--host", "0.0.0.0", "--port", "8080"]

{
  "textPayload": "Traceback (most recent call last):\n  File \"/usr/local/bin/uvicorn\", line 8, in <module>\n    sys.exit(main())\n  File \"/usr/local/lib/python3.10/site-packages/click/core.py\", line 1157, in __call__\n    return self.main(*args, **kwargs)\n  File \"/usr/local/lib/python3.10/site-packages/click/core.py\", line 1078, in main\n    rv = self.invoke(ctx)\n  File \"/usr/local/lib/python3.10/site-packages/click/core.py\", line 1434, in invoke\n    return ctx.invoke(self.callback, **ctx.params)\n  File \"/usr/local/lib/python3.10/site-packages/click/core.py\", line 783, in invoke\n    return __callback(*args, **kwargs)\n  File \"/usr/local/lib/python3.10/site-packages/uvicorn/main.py\", line 425, in main\n    run(app, **kwargs)\n  File \"/usr/local/lib/python3.10/site-packages/uvicorn/main.py\", line 447, in run\n    server.run()\n  File \"/usr/local/lib/python3.10/site-packages/uvicorn/server.py\", line 68, in run\n    return asyncio.run(self.serve(sockets=sockets))\n  File \"/usr/local/lib/python3.10/asyncio/runners.py\", line 44, in run\n    return loop.run_until_complete(main)\n  File \"/usr/local/lib/python3.10/asyncio/base_events.py\", line 649, in run_until_complete\n    return future.result()\n  File \"/usr/local/lib/python3.10/site-packages/uvicorn/server.py\", line 76, in serve\n    config.load()\n  File \"/usr/local/lib/python3.10/site-packages/uvicorn/config.py\", line 448, in load\n    self.loaded_app = import_from_string(self.app)\n  File \"/usr/local/lib/python3.10/site-packages/uvicorn/importer.py\", line 24, in import_from_string\n    raise exc from None\n  File \"/usr/local/lib/python3.10/site-packages/uvicorn/importer.py\", line 21, in import_from_string\n    module = importlib.import_module(module_str)\n  File \"/usr/local/lib/python3.10/importlib/__init__.py\", line 126, in import_module\n    return _bootstrap._gcd_import(name[level:], package, level)\n  File \"<frozen importlib._bootstrap>\", line 1050, in _gcd_import\n  File \"<frozen importlib._bootstrap>\", line 1027, in _find_and_load\n  File \"<frozen importlib._bootstrap>\", line 992, in _find_and_load_unlocked\n  File \"<frozen importlib._bootstrap>\", line 241, in _call_with_frames_removed\n  File \"<frozen importlib._bootstrap>\", line 1050, in _gcd_import\n  File \"<frozen importlib._bootstrap>\", line 1027, in _find_and_load\n  File \"<frozen importlib._bootstrap>\", line 1006, in _find_and_load_unlocked\n  File \"<frozen importlib._bootstrap>\", line 688, in _load_unlocked\n  File \"<frozen importlib._bootstrap_external>\", line 883, in exec_module\n  File \"<frozen importlib._bootstrap>\", line 241, in _call_with_frames_removed\n  File \"/app/./challenge/__init__.py\", line 1, in <module>\n    from challenge.api import app\n  File \"/app/./challenge/api.py\", line 9, in <module>\n    from challenge.model import DelayModel\n  File \"/app/./challenge/model.py\", line 7, in <module>\n    from xgboost import XGBClassifier\nModuleNotFoundError: No module named 'xgboost'",
  "insertId": "66537deb000d6795627223e2",
  "resource": {
    "type": "cloud_run_revision",
    "labels": {
      "revision_name": "fastapi-model-00004-8hg",
      "location": "us-central1",
      "service_name": "fastapi-model",
      "configuration_name": "fastapi-model",
      "project_id": "challenge-mle"
    }
  },
  "timestamp": "2024-05-26T18:22:35.878485Z",
  "severity": "ERROR",
  "labels": {
    "instanceId": "00f46b92854722e97782eabce2ca123a9883d315d6c7e4218161cbb9e366c094b2774ea694694cba9fbf511d28919eb8f94a1ee88df51f25ccdbf1a7aaef2b9d00"
  },
  "logName": "projects/challenge-mle/logs/run.googleapis.com%2Fstderr",
  "receiveTimestamp": "2024-05-26T18:22:35.881188747Z",
  "errorGroups": [
    {
      "id": "CPK33IqA79XrdA"
    }
  ]
}