
# Summary and home assignment 
- We went on quite a ride this week 
- You learned to do many things 
- Also your cluster account should be looking good
- The last thing we want to do is gain access to the course shared space

---

# Collaborating with version control 

- While I advocated for using GitHub as collaboration hub in reality most of Neuroscience research done relies on large data-sets that contain sensitive information 
- Therefore, in this course we will be using a shared space on the cluster rather than a collaborative repository  
- It is also worth noting that using the collaborative part of GitHub requires additional work that I felt was not necessary for our needs
- However for those who wish to explore this option further the web is full with examples and you have all the right functionality to either join or host such a repository


---

# Collaborating using a shared local space  

- What we will use instead is a shared space on our scratch disk
- Please open a terminal on the [HPC Jupyter-hub](https://jupyter-hpc.gwdg.de/)
- Now cd into `/scratch/systemAI/course_2020`
- This will be our course space 

---

# Creating a symbolic link 

- The shared space is empty now 
- But it contains some hierarchy
- We will start by creating a symbolic link to the folder on your home directory  

---

# Creating a symbolic link 

- A symbolic link, also known as a symlink or soft link, is a special type of file that points to another file or directory
- Syntax is dead easy 

```bash
ln -s {source-path} {symbolic-name}
``` 
- Goto your home directory create a folder called learning 
- And in it create a symlink to the course material 
- Go to the next slide **AFTER** you did this 

---

# Creating a symbolic link 

- In order to use the jupyterhub with our shared space we will make a soft link to the course folder that will be accessible on the web interface
- Jupyterhub has access to the jupyterhub-gwdg folder in your home directory 
- Creating a soft link to our course folder makes the entire folder tree accessible to the web interface 

```bash
cd ~ && mkdir ~/jupyterhub-gwdg/learning
source=/scratch/systemAI/course_2020
target=~/jupyterhub-gwdg/learning/systemAI_2020
ln -s  $source $target
```

---

# Home assignment 

1. Download (using <a href="#/rsync">rsync</a>) the pdf in the papers folder to your personal computer and read it before our meeting. 
1. Goto the week_00 folder under notebook and create a folder with your name 
1. Upload (using <a href="#/rsync">rsync</a>) to that folder your portrait as 
1. Finally navigate to your folder and use the web interface to create a text file named <yourname>_profile.md

---

# _profile.md

- In this file I want you to add the following information 
    1. Introduce yourself 
        1. Name 
        1. Hobbies
        1. What is your favorite movie.
    1. Why did you choose to enroll in this program
    1. What is the most important skill you hope to learn in this course.
    1. What is your academic background 


---

# That's it for this HPC boot camp week 
<div class="centered_image">
<img src="http://phdcomics.com/comics/archive/phd102402s.gif">
</div>

# Next week will be covering 
# Python and Jupyter nootbook




