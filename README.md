# Towards reproducible builds with Docker development containers

## Create your own development environment

* create you own development environment by installing all required packages, libraries, dependencies, environmental variables, paths, and anything required to build your project by adding stuff into the `devctr/Dockerfile`
    * the provided Dockerfile already contains `cuda:11.8.0` toolchain and `python3`
* build your own development environment: `./devtool build_devctr`
* (OPTIONALLY) If you want to share your development environment with other people:
    * register on [Dockerhub](https://hub.docker.com/)
    * change `DOCKER_HUB_USERNAME` in `devtool` to your username
    * push changes with `docker push <your username>:6_s894_finalproject_devctr:latest`

Now you can build things inside this development container anywhere, without "it works on my machine" issues anymore and without installing anything (which might be very complicated and messy sometimes) on your physical machine (host).

## Building your projects and (OPTIONAL) executing via Telerun

* put your project you want to build inside `src` folder
* write what should be called to build your project in `src/build.sh`
    * if you want to add some test input files along your binaries, write instructions in `build.sh` as well
* (OPTIONAL) write what should be executed on the GPU server in `src/run.sh`
* build it: `./devtool build_project`
* the output can be found in `build` folder and all together in `build.tar`
* (OPTIONAL) `build.tar` is shippable for execution via `Telerun`:
    * `python3 <path_to_telerun.py> submit build.tar`
    * Telerun will execute your `run.sh`
