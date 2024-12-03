Steps
0. symlink telerun to urs/bin or usr/local/bin (if you're on a mac lol like a loser)
    - ln -s /path_to_telerun/telerun.py /usr/local/bin/telerun
1. Start docker OR orbstack
2. run src/generate_attention_data.py
3. run ./devtool test
4. if you encounter issues, contact khm@mit.edu


To run the actual build:
./devtool build_devctr
./devtool build_project
telerun submit build.tar
