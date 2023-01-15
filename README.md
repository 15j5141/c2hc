# How to
## create records.json
1. download and unarchive Project_CodeNet
    ```sh
    wget https://dax-cdn.cdn.appdomain.cloud/dax-project-codenet/1.0.0/Project_CodeNet.tar.gz
    tar -zxf Project_CodeNet_full.tar.gz
    mv ./Project_CodeNet in/Project_CodeNet
    ```

1. Only once execute.
    ```sh
    npm i
    ```
1. create records.json from "Project_CodeNet"
    ```sh
    node getPathList.js
    ```

## execute analyze
1. execute python
    ```sh
    python manager.py
    ```
