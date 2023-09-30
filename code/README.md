## Clonning the GitHub repository

```bash

git clone https://github.com/Malware-Hunter/WRSeg23-FS3E-v2.git

cd code

```

## Building and running your own Docker :whale: image


1. Installing Docker and building your image
```bash

sudo apt install docker docker.io

docker  build  -t  fs3e:latest  .

```

2. Starting a Docker container in **persistent** or **non persistent** mode.

**Non persistent mode**: output files will be deleted when the container finishes execution.
```bash

docker  run  -it  fs3e

```
**Persistent mode**: output files will be saved and avaliable at the current directory.
```bash

docker run -v $(readlink -f .):/fs3e -it fs3e

```


## :memo: Running it in your Linux

Installing requirements
~~~sh
pip install -r requirements.txt
~~~

## :pushpin: Available Arguments:

```
usage: fs3e.py run [-h] (--ft-types TYPE [TYPE ...] | --all-ft-types) (--fs-methods METHOD [METHOD ...] | --all-fs-methods) -d DATASET
                   [DATASET ...] [-c CLASS_COLUMN] [--verbose] [--output OUTPUT]

Optional Arguments:
  -h, --help            show this help message and exit
  --ft-types TYPE [TYPE ...]
                        Methods of SELECTED Features Types. Choices: ['api', 'permission']
  --all-ft-types        Methods of ALL Features Types
  --fs-methods METHOD [METHOD ...]
                        Run Selected Methods. Choices: ['abc', 'fsdroid', 'jowmdroid', 'linearregression', 'mt', 'rfg', 'semidroid', 'sigapi',
                        'sigpid']
  --all-fs-methods      Run ALL Methods
  -d DATASET [DATASET ...], --datasets DATASET [DATASET ...]
                        One or More Datasets (csv Files). For All Datasets in Directory Use: [DIR_PATH]/*.csv
  -c CLASS_COLUMN, --class-column CLASS_COLUMN
                        Name of the class column. Default: "class"
  --verbose             Show More Run Info.
  --output OUTPUT       Output File Directory. Default: ./results
```

## :gear: Environments

FS3E has been tested in the following environments:

**Ubuntu 20.04**

- Kernel = `Linux version 5.4.0-120-generic (buildd@lcy02-amd64-006) (gcc version 9.4.0 (Ubuntu 9.4.0-1ubuntu1~20.04.1)) #136-Ubuntu SMP Fri Jun 10 13:40:48 UTC 2022`
- Python = `Python 3.8.10`

## Collaboration & Review
## Collaboration
The following individuals collaborated in this project:

- Lucas
- Vanderson
- Nicolas
- Luiza
- Savio
- Joner

## Review

Review
The work was reviewed by:

- Diego
- Vanderson

This was a collaborative process, where each individual contributed to the implementation of one or more methods. All implementations were reviewed to ensure fidelity to the original.

Link to Papers
- FSDROID: https://link.springer.com/article/10.1007/s11042-020-10367-w
- SEMIDROID: Link
- LINEARREGRESSION: Link
- RFG: Link
- ABC: Link
- JOWNDROID: Link
- MT: Link
- SIGPID: Link
- SIGAPI: Link

