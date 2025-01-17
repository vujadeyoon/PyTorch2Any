# Template-Container


## Table of contents
1. [Notice](#notice)
2. [Clone the git repository](#clone_repo)
3. [Container](#container)
4. [Run](#run)
5. [Reference](#ref)
6. [Todo](#todo)


## 1. Notice <a name="notice"></a>
- I recommend that you should ignore the commented instructions with an octothorpe, #.


## 2. Clone the git repository <a name="clone_repo"></a>
```bash
(HOST) $ git clone --recursive ${URL_GIT_REPO}
(HOST) $ cd ./vujade-python/ && bash ./script/bash_setup_vujade.sh && cd ../
```


## 3. Container <a name="container"></a>
### 1. Build container
```bash
(HOST) $ bash ./script/host/bash_build_container.sh "vujadeyoon:latest" "vujadeyoon" "vujadeyoon@gamil.com" "/home/dev/proj" "$(pwd)/Containerfile"
```
### 2. Run container
```bash
(HOST) $ bash ./script/host/bash_run_container.sh "vujadeyoon:latest" "$(pwd):/home/dev/proj" "10001:11001" "/home/dev"
```


## 4. Run <a name="run"></a>
```bash
(Container) $ bash ./script/container/bash_run.sh
```


## 5. Reference <a name="ref"></a>


## 6. Todo <a name="todo"></a>
