# PFM-1 Mine Detection and Tracking

![PFM-1 mine](./pfm-1.png)

## About the Project

### Getting Started

To get started

1. Clone the Repository

```bash
  git clone https://github.com/Lihemen/pfm-mine-detection-algo
```

2. Create Virtual Environment

```bash
  python3 -m venv venv
```

or (python version 3.11.x)

```bash
  python -m venv venv
```

3. Enable Virtual Environment

Mac or Linux

```bash
  source ./venv/bin/activate
```

Windows Bash

```bash
  source venv/Scripts/activate
```

Windows Powershell

```bash
  Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process -Force
```

```bash
  .\venv\Scripts\activate.ps1
```

4. Install Dependencies

```bash
  pip install -r requirements.txt
```

5. Running the project

Open your terminal and write the command

```bash
  python3 main.py
```

or

```bash
  python main.py
```

If in VSCode, you can click the Play button that appears on the right when in a python file. Choose the environment of the created venv and run.

5. Working with Git

- Create a branch

To prevent working directly on the main branch, you can create a copy based off the latest changes in the main branch

```bash
  git checkout -b <branch-name> <e.g git checkout -b sara_branch>
```

Once this branch has been merged into main, it can be deleted safely, all changes would be preserved in main.

- Commit your changes

This add all the files (both new and modified) to the current working directory

```bash
  git add .
```

This saves the current working directory ready to be sent.

```bash
  git commit -am "Commit message" <e.g git commit -am "Updated plotting">
```

- Save your changes

This sends the changes to github for everyone to see

```bash
  git push -u origin <branch_name> <e.g git push -u origin sara_branch>
```

- Get the latest changes

This moves the cursor back to the main branch

```bash
  git checkout main
```

This fetches and merges the latest changes

```bash
  git pull
```

- Once you're back on the main branch you can delete your old branch and create a new branch

This deletes the branch

```bash
  git branch -D <old_branch_name>
```

This creates a new branch

```bash
  git checkout -b <new_branch_name>
```

The names can be the same, provided this command is executed when you're on the main branch.

5. Train the Model (optional)

6. Run the Model

7. Visualize Data

### Running Label Studio

For further labelling, run the following command in the terminal

```bash
  label-studio start
```

- Create an account using any email and password
- Upload new images to the project
- Draw bounding boxes
- Export yolov8 format when done.

### Milestones

### Challenges

### Improvements

- Collect more diverse training data (different lighting, angles, seasons)
- Use synthetic data generation for rare scenarios

Authors: [Hemense Lan](hemense.lan@student.uhasselt.be), [Simon Derwael](simon.derwael@student.uhasselt.be), [Sara Reihani](sara.reihani@student.uhasselt.be)
