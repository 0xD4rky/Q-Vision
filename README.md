# Q-Vision
Analytics of quantization on vision models 


Q-Vision/
├── README.md
├── LICENSE
├── setup.py
├── requirements.txt
├── qvision/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── model_loader.py
│   │   ├── quantizer.py
│   │   ├── optimizer.py
│   │   ├── evaluator.py
│   │   ├── exporter.py
│   │   └── hardware_support.py
│   ├── interfaces/
│   │   ├── __init__.py
│   │   ├── cli.py
│   │   └── gui.py
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── data_loader.py
│   │   ├── transforms.py
│   │   └── augmentation.py
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── dashboard.py
│   │   └── plots.py
│   ├── deployment/
│   │   ├── __init__.py
│   │   ├── edge_devices.py
│   │   ├── mobile.py
│   │   ├── cloud.py
│   │   └── containerization.py
│   ├── examples/
│   │   ├── __init__.py
│   │   ├── quantize_yolov5.py
│   │   ├── quantize_faster_rcnn.py
│   │   └── ...
│   ├── tests/
│   │   ├── __init__.py
│   │   ├── test_model_loader.py
│   │   ├── test_quantizer.py
│   │   ├── test_optimizer.py
│   │   └── ...
│   └── utils/
│       ├── __init__.py
│       ├── logging.py
│       └── config.py
├── docs/
│   ├── index.md
│   ├── installation.md
│   ├── user_guide.md
│   ├── api_reference.md
│   └── tutorials/
│       ├── quantization_basics.md
│       └── advanced_quantization.md
├── scripts/
│   ├── install.sh
│   ├── run_tests.sh
│   └── deploy_docker.sh
└── .github/
    ├── ISSUE_TEMPLATE.md
    ├── PULL_REQUEST_TEMPLATE.md
    └── workflows/
        └── ci.yml
