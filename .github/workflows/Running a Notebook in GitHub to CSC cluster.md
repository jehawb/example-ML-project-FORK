# Running a Notebook in GitHub to CSC cluster

## Manual requirements (to be automated?)

### SSH key pair to the remote cluster

You need a keypair **without the passphrase** (or empty passphrase) setup with the remote cluster's server.

### Notebook in GitHub

1. The [Notebook](https://github.com/OSS-MLOPS-PLATFORM/oss-mlops-platform/blob/main/tutorials/demo_notebooks/demo_pipeline/demo-pipeline.ipynb) needs to be named `demo-pipeline.ipynb`.
2. For the Notebook to work it also needs the [.yaml files](https://github.com/OSS-MLOPS-PLATFORM/oss-mlops-platform/tree/main/tutorials/demo_notebooks/demo_pipeline/components) in to a folder named `components/` located on the same file level as the Notebook itself.
    - ***NB!*** The *.yaml files are hidden* in GitHub's file manager!
3. You may want to rename the run from `demo-run` to something more descriptive in the Submit run part of the Notebook.

### GitHub secrets

The GitHub actions .yaml expects three **repository secrets** to be setup for the repo (repo settings > Secrets and variables > Actions > Repository secrets)

1. `REMOTE_CSC_CLUSTER_SSH_PRIVATE_KEY`: contents of the private key file located in your computer's `.ssh/` folder e.g.
    
    ```
    -----BEGIN RSA PRIVATE KEY-----
    MIIBOgIBAAJBAKj34GkxFhD90vcNLYLInFEX6Ppy1tPf9Cnzj4p4WGeKLs1Pt8Qu
    KUpRKfFLfRYC9AIKjbJTWit+CqvjWYzvQwECAwEAAQJAIJLixBy2qpFoS4DSmoEm
    ...
    -----END RSA PRIVATE KEY-----
    ```

2. `REMOTE_CSC_CLUSTER_SSH_IP`: e.g. 123.123.123.123
3. `REMOTE_CSC_CLUSTER_SSH_USERNAME`: e.g. username

### Action .yml

Needs to be placed in a folder system `/.github/workflows/` in the repo's root for GitHub to recoqnize it as GitHub action.
As this is connecting to the remote cluster it should be in the staging or the production branch.

## GitHub Actions .yml file

```
name: Run Jupyter Notebook connected to remote server

on:
  push:

jobs:
  run-notebook:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.8
        
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install jupyter
        
    - name: Connect to remote server and run the Notebook
      env:
        REMOTE_CSC_CLUSTER_SSH_PRIVATE_KEY: ${{ secrets.REMOTE_CSC_CLUSTER_SSH_PRIVATE_KEY }}
        REMOTE_CSC_CLUSTER_SSH_IP: ${{ secrets.REMOTE_CSC_CLUSTER_SSH_IP }}
        REMOTE_CSC_CLUSTER_SSH_USERNAME: ${{ secrets.REMOTE_CSC_CLUSTER_SSH_USERNAME }}
      run: |
        # Create the .ssh directory if it doesn't already exist to store SSH keys
        mkdir -p ~/.ssh
        
        # Write the private SSH key to a file and set its permissions to be read/write for the owner only
        echo "$REMOTE_CSC_CLUSTER_SSH_PRIVATE_KEY" > ~/.ssh/id_rsa
        chmod 600 ~/.ssh/id_rsa

        # Add the remote server's host key to the known_hosts file
        ssh-keyscan -H $REMOTE_CSC_CLUSTER_SSH_IP >> ~/.ssh/known_hosts
        
        # Establish an SSH connection to the remote server in the background, forwarding port 8080
        ssh -i ~/.ssh/id_rsa -L 8080:localhost:8080 $REMOTE_CSC_CLUSTER_SSH_USERNAME@$REMOTE_CSC_CLUSTER_SSH_IP -N &
       
        # Execute the Jupyter Notebook and overwrite the existing file with the results
        jupyter nbconvert --execute --to notebook --inplace notebooks/demo-pipeline.ipynb
        
```

## TODO

- Automate things
    - SSH key pair creation
    - Adding the secrets to GitHub
    - The run is pushed to the cluster every time the repo is pushed to -> maybe track only certain folders for changes
- Rename things
    - Currently names mention CSC which propably isn't going to be always the place of the remote cluster -> rename to something more generic but descriptive
    - Also demo-pipeline.ipynb file name is used for the Notebook currently
    - IP and Usernames setup currently is CSC spesific, maybe change it to IP for the known hosts file and connection address for the SSH
      - Recheck if adding the IP to the known hosts is a must-have
- Correct place for the secrets 
    - GitHub offers: Env. secrets / Repo secrets / Env. vars / Repo vars
    - Currently using Repo secrets for all three secrets
- Using something else than plain Ubuntu
    - Some other container might have the Python and Jupyter already installed allowing to skip the dependency installs
    - Also should some spesific Ubuntu be used instead of latest?
