# Testing

## Fast Checks

These checks do not load model weights and are suitable before pushing changes:

```bash
python -m unittest discover
python -m py_compile main.py src\utils\config.py src\detector\detector.py src\ocr\recognizer.py src\pipeline\lpr_pipeline.py src\database\access_db.py src\controller\barrier_controller.py demo\setup_whitelist.py demo\run_demo.py demo\test_yolov5_weights.py
```

The unit tests verify:

- config loading;
- whitelist determinism;
- text normalization;
- database access logging.

## End-to-End Demo

```bash
python demo/run_demo.py
```

Expected result:

```text
Status:             PASS
```

The script writes `data/videos/demo_output.mp4` and creates
`data/access_list.db`. Both files are generated artifacts and are ignored by
git.

For a slower full-frame run:

```bash
python demo/run_demo.py --seconds 0 --process-every 1
```
