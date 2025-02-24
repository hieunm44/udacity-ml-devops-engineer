# Build an ML Pipeline for Short-Term Rental Prices in NYC

## Tổng quan
ML Pipeline project có các thành phần:
* `main.py`: File chính cho toàn bô pipeline
    ```python
        # Dùng hydra decorator để tự động xác định file config là config.yaml
        @hydra.main(config_name='config')
        def go(config: DictConfig):
    ```
        
* File `MLproject` để define sẽ làm gì khi chạy lệnh mlflow:
    ```python
    name: nyc_airbn
    conda_env: conda.yml
    
    entry_points:
        main:
        parameters:
    
            steps:
            description: Comma-separated list of steps to execute (useful for debugging)
            type: str
            default: all
    
            hydra_options:
            description: Other configuration parameters to override
            type: str
            default: ''
    
        command: "python main.py main.steps=\\'{steps}\\' $(echo {hydra_options})"
    ```
    
* `config.yaml`: Chứa các thiết lập cho pipeline
    ```yaml
    main:
        components_repository: "https://github.com/udacity/build-ml-pipeline-for-short-term-rental-prices#components"
        project_name: nyc_airbnb
        experiment_name: development
        steps: all
    etl:
        sample: "sample1.csv"
        min_price: 10 # dollars
        max_price: 350 # dollars
    data_check:
        kl_threshold: 0.2
    modeling:
        test_size: 0.2
        val_size: 0.2
        random_seed: 42
        stratify_by: "neighbourhood_group"
        max_tfidf_features: 5
        random_forest:
        n_estimators: 200
        max_depth: 50
        min_samples_split: 4
        min_samples_leaf: 3
        n_jobs: -1
        criterion: mae
        max_features: 0.5
        oob_score: true
    
    ```
    
* `conda.yml`: chứa cá thư viện conda cần cho pipeline
    ```python
    name: components
    channels:
        * conda-forge
    dependencies:
        * mlflow=1.14.1
        * pip=20.3.3
        * pip:
            * wandb==0.10.31
            * pyyaml==5.3.1
            * hydra-core==1.0.6
            * numpy==1.23.0
    ```
    
* Folder `src` chứa các steps trong pipleline. Gồm bốn subfolders:
`src/basic_cleaning`, `src/data_check`, `src/eda`, `src/train_random_forest`.
* Các steps khác đc pre-implemented trong folder components, gồm ba subfolders:
`components/get_data`, `components/test_regression_model`, `components/train_val_test_split`.
* Mỗi subfolder đều có file MLProject để define sẽ làm gì khi chạy lệnh mlflow trong file `main.py` ở root folder.

`mlflow`: Platform cho ML pipeline
* Chạy một step trong pipeline
    
    ```bash
    mlflow run src/step_name -P parameter1=1 -P parameter2=2 -P parameter3="test"
    ```
    
* Chạy toàn bộ pipeline
    
    ```bash
    mlflow run .
    ```
    
* Chạy step cụ thể
    
    ```bash
    mlflow run . -P steps=download, basic_cleaning
    ```
    
* Có thể thay đổi parameter trong config bàng `hydra_options` parameter:
    
    ```bash
    mlflow run . \
        -P steps=download,basic_cleaning \
        -P hydra_options="modeling.random_forest.n_estimators=10 etl.min_price=50"
    ```
    
* Có thể chạy pre-implemented components. VD lệnh sau truy cập vào:
[https://github.com/udacity/build-ml-pipeline-for-short-term-rental-prices#components](https://github.com/udacity/build-ml-pipeline-for-short-term-rental-prices#components)
folder `components/get_data`, version `main`, các parameters đc defined trong file `MLproject`.
    
    ```bash
    _ = mlflow.run(
                    f"{config['main']['components_repository']}/get_data",
                    "main",
                    version='main',
                    parameters={
                        "sample": config["etl"]["sample"],
                        "artifact_name": "sample.csv",
                        "artifact_type": "raw_data",
                        "artifact_description": "Raw file as downloaded"
                    },
                )
    ```
        

## Các bước chạy pipeline
### EDA
* Download data: Bước này đã đc provided là component, script chính là `components/get_data/run.py`, sẽ download môt data file (thực chất là file `sample1.csv`), đc define trong là `sample` trong `config.yaml`. Ở file `run.py` ta phải define các parameters bằng ArgumentParser. Trong file `main.py`, ở bước chạy mlflow ta sẽ chỉ rõ giá trị các parameters này.
* EDA
    ```bash
    mlflow run src/eda
    ```
    
    Sẽ mở jupyter notebook.
    

### Data cleaning
Script chính là file `src/basic_cleaning/run.py`, sẽ tải data về từ W&B, tạo ra `clean_sample.csv`, rồi upload nó lên W&B. Trong file `main.py` ở root folder, chạy mlflow ở bước `basic_cleaning`.

### Data testing
Script chính là file `src/data_check/test_data.py`, chứa một số hàm test. Trong file `main.py` ở root folder, chạy mlflow ở bước `data_check`.

### Data splitting
Ta dùng pre-implemented component ở folder `component/train_val_test_split`, chạy bằng mlflow trong file `main.py`:
```python
_ = mlflow.run(
                f"{config['main']['components_repository']}/train_val_test_split",
                'main',
                version='main',
                parameters={
                    'input': 'clean_sample.csv:latest',
                    'test_size': config['modeling']['test_size'],
                    'random_seed': config['modeling']['random_seed'],
                    'stratify_by': config['modeling']['stratify_by']
                },

            )
```
    

### Train Random Forest
Script chính là file `src/train_random_forest/run.py`, chứa một số hàm test. Trong file `main.py` ở root folder, chạy mlflow ở bước `train_random_forest`.

### Optimize hyperparameters
Chạy pipeline khi mà thay đổi các hyperparameters của Random Forest bằng `hydra_options`:    
```python
mlflow run . \
-P steps=train_random_forest \
-P hydra_options="modeling.random_forest.max_depth=10,50,100 modeling.random_forest.n_estimators=100,200,500 -m"
```
    
### Select the best model
Chọn best model trên W&B

### Test
* Đc chạy bằng provided component ở folder `component/test_regression_model`.
* Bước này ko đc chạy mặc định trong pipeline. Cần phải chạy nó explicitly:
    
    ```python
    mlflow run . -P steps=test_regression_model
    ```
    
### Visualize the pipeline
Release  trên Github

### Train the model on a new data sample
Chạy pipeline trên data mới `sample2.csv`:
```python
mlflow run https://github.com/[your github username]/build-ml-pipeline-for-short-term-rental-prices.git \
                -v [the version you want to use, like 1.0.0] \
                -P hydra_options="etl.sample='sample2.csv'"
```