# Basics of Version Control (VC)

- In this section we will cover what is Version Control
- Why we need it 
- We will also learn how to install git as our VC system
- And how to create our first repository to store important configuration files 
- As always [here](https://blog.scottlowe.org/2015/01/14/non-programmer-git-intro/) is a good introduction to git from a non-programmer point of view 

//slide//

## Why do we need a VC system

- Up until now we covered how to work on the cluster, and how to move files from and to the cluster 
- However, this kind of behavior requires you to constantly keep both sides (cluster and home computer) in sync
- If you also have a Desktop computer tracking what changed and when becomes very hard
- When we consider working on a project with collaborators this is impossible 
- Luckily enough this problem isn't uncommon and it is simple to solve using a process called version control

//slide//

## What does a VC do?

-  Version control is a system that simplifies managing and tracking changes made to content of a project
- In this course we will learn how to use a tool called global information tracker (git for short)
- And will rely on the service of [GitHub](https://en.wikipedia.org/wiki/GitHub) as our online repositories providers. 
- This is because it is free of charge and easy to learn and use.
- If you don't have an account please press on [join GitHub](https://github.com/join) and follow the instructions   
- Also [install GitHub](https://gist.github.com/derhuerst/1b15ff4652a867391f03) on your local computer.


//slide//

## What is a repository?

- A repository is a central location in which data is stored and managed.
- Git can be viewed as a method to control, manage and document a projects evolution
- Project evolution is composed of
    - Files (i.e. content)
    - Directories (i.e. hierarchy)
    - Changes across time (i.e. history)
    - Users (i.e. contribution)
- When you create a new repository a hidden dot folder named .git is created where that data is stored. 


//slide//

## Is git installed on the Cluster?

- Well the short answer is no!
- We will have to install it ourselves
- Unfortunately this is not as simple as setting it on your personal computer 
- This is because we don't have administrator rights on the cluster 
- However we can use a process called compilation that will allow us to use open-source code to create a functional program
- As most of you are lacking advanced programmatic skills (at the moment) I made the instructions as clear as I could.  
- If you follow them to the letter I'm sure we will manage 


//slide//

## Log onto the Cluster

- In order to install things we need to be on the cluster
- So open a terminal on your personal computer
- And use ssh to connect to the cluster 
- <a href="#/ssh_login">This is how we did it previously</a>



//slide//

## Download git

- We will create a temporary folder on our home directory
- Then we will use `wget` to download a compressed snapshot of the recent version of the git program (tar ball)
- We will use tar (a compression program similar to zip) to uncompress the tar ball
- Finally we will go in the directory

```bash
mkdir ~/src # create a src dir 
cd ~/src # go into it
# Download a version of GitHub
wget  https://github.com/git/git/archive/v2.26.2.tar.gz
tar -xzvf * # Expand the tar file (tarball)
cd git-2.26.2/ # And goto the folder 
```

//slide//

## Compiling git on the cluster

- Compiling a program can be a frustrating experience 
- However, in this case this is highly unlikely 
- We use a process that is called `make` which will build the software specifically for the install location you want 
- In our case we are compiling to the bin folder in your home directory 
- The following `make install` will install the program in that location 
- Notice the use of double & to chain commands 
- Running this will take a while and spit many words 

```bash
make && make install
git --version # confirm that this worked 
```
//slide//

## Generate a new rsa key 

- Now we need to generate a new rsa key 
- Previously we generated an rsa key from our home computer to simplify the login process to the cluster 
- We will now use the same process to simplify interaction to and from GitHub

```bash
ssh-keygen -t rsa -b 4096 -C <the_email@you_used_on_github.com>
eval "$(ssh-agent -s)" # Start the ssh-agent in the background.
ssh-add ~/.ssh/id_rsa # Add your SSH private key to the ssh-agent
cat ~/.ssh/id_rsa.pub # Print your key and copy it
```

//slide//

## Setting up ssh key access to Github

Now you need to go over to this [LINK](https://help.github.com/en/github/authenticating-to-github/adding-a-new-ssh-key-to-your-github-account) and follow the instructions to the letter. 


```bash
cd ~ 
rm -rf src # Delete the src folder 
# Add some information to you account 
git config --global user.name "Your git user Name" 
git config --global user.email "<your mail>@mpi-mail.mpg.de"
# Open ~/.ssh/config
nano ~/.ssh/config
# Copy the following line to the TOP of the file
IgnoreUnknown AddKeysToAgent,UseKeychain
# Now check your connection to your GitHub repository 
ssh -T git@github.com
``` 
//slide//

## Setting up cluster access to Github

- We are almost ready to use git and GitHub
- We only need to setup the cluster to enable us to create and delete repositories from the command line 
- To do this we need to generate a token which is a way to extend normal passwords by associating different permissions for each token


//slide//

## Creating an api token on Github

- Please follow this [link](https://help.github.com/en/github/authenticating-to-github/creating-a-personal-access-token-for-the-command-line
) to create a personal access token for the command line
- Or alternatively go [here](https://github.com/settings/tokens) if you don't need the tutorial to generate a new token
- Make sure to tick the following boxes:
    - `Delete repositories` 
    - `repo` 

//slide//

## A small detour to talk about security and permissions 

- The token you just created is a scary thing as it allows anyone with it to delete your repositories 
- You should treat it and any password on the cluster with care 
- One way to do that is to separate between the configuration files and sensitive information 
- In linux any file or folder will have permission settings 

//slide//

## Why are permissions important and useful?

- Permissions control who can have access to a file or folder 
- And more importantly what kind of access is granted 
- A good example is experimental raw data 
- We will want many people to have read access
- But other than the creator of the data no one should have write access 


//slide//

## Basics of permissions 

- Using `ls -l ~/.bashrc` give the following :
![](/img/Tutorial_bashrc_permissions.png)
- Let's unpack the bottom line from right to left


| :-- | :-- |
| `/usr/users/esoreq/.bashrc` | full path name |
| `May 12 08:43` | The **last** modification time |
| `172` | File size in bytes |
| `MLNP` | Group name |
| `esoreq` | Owner name |
| `1` | Number of links | 
| `-rw-------` | Permissions |      
| | |

//slide//

## Basics of permissions 

- Permission settings are grouped in a string of characters (-, r, w, x) 
- The characters r, w, and x stand for **read**, **write**, and **execute**.
classified into four sections:
    - <span style="color:GOLD">-</span>rwxrwxrwx - File type has three possibilities:
        - regular file (–)
        - a directory (d)
        - or a link (i)
    - -<span style="color:GOLD">rwx</span>rwxrwx - File permission of the user (owner)
    - -rwx<span style="color:GOLD">rwx</span>rwx - File permission of the owner’s group
    - -rwxrwx<span style="color:GOLD">rwx</span> - File permission of other users

//slide//

## Changing permissions 

- Any Linux user, will at some point need to modify the permission settings of a file/directory.
- This can be done using the `chmod` command.
- The basic syntax is:

```bash
chmod [permission] [file_name]
# one way that is the simplest to remember 
chmod u=rwx,g=rwx,o=rwx [file_name]
```
- rwx stand for r(ead),w(rite) and (e)x(ecute) 
- and ugo stand for u(ser), g(roup) and o(ther)
//slide//

## Changing permissions using the numeric mode 

- Instead of letters, the numeric format represents privileges with numbers:
    - r(ead) has the value of 4
    - w(rite) has the value of 2
    - (e)x(ecute) has the value of 1
    - no permission has the value of 0
- Privileges are summed up and depicted by one number. 
- The permission is compressed to three (3) numbers 
- Each representing the summation of privileges for each category (user, group, owner).


//slide//

## Changing permissions using the numeric mode 

- Therefore, the possibilities are:
    - 7 = read, write, and execute permission
    - 6 = read and write privileges
    - 5 = read and execute privileges
    - 4 = read privileges
- And in practice the basic syntax changes to:



```bash
chmod 750 [file_name] # -rwxr-x---
chmod 400 [file_name] # -r--------
```



//slide//

## Back to practicals

- To have our passwords protected we will create a personal folder 
- Make sure only we have access to it 
- Create a file called passwords 
- Append our github token into it 
- And add a line to our bashrc file to source this variable on login

```bash
GIT_TOKEN=<the token>
mkdir ~/personal && cd ~/personal
chmod 700 ~/personal # Only you will have access
touch passwords
echo "GIT_TOKEN=$GIT_TOKEN" >> passwords
echo ". ~/personal/passwords" >> ~/.bashrc
. ~/.bashrc
```
- If all worked well you should have a working git on the cluster 

//slide//

## Creating your first repository  

- Let's create a repository that stores the various configuration files we just created
- This is not solely for the experience of creating a repository, 
- It is useful to have when moving from one position to another  
- And allows you to share your work with other people in your lab 


//slide//

## Initializing a git folder 

- The `git init` function is used to create the hidden `.git` folder within your repository 
- When we run `git init [foldername]` we create a new folder with the `.git` folder inside 
- Alternatively we can initialize an existing folder by going into the folder and running `git init` without any other options 

```bash
git init $HOME/dotfiles/ # make a git initialized folder 
```

//slide//

## Add some content

- We will need to add some content (i.e. files and folders) into our repository
- Good practice is to always include a Readme.md markdown file 
- Markdown is a rich text syntax that we will learn next week

```bash
# copy over your dot files
cp .bash_aliases .bashrc .profile .bash_profile ~/dotfiles/
ls -af dotfiles/.??*  # confirm that they were copied 
cd ~/dotfiles # goto the folder 
touch Readme.md # create a readme file (good practice)
echo "A repository for dot configuration files" >> Readme.md
```

//slide//

## Create a remote repository

- We now have a local folder with some files inside 
- We want to have a matching repository on the GitHub site
- We could just use the web interface to manually create our dotfiles repository
- An alternative way would be to create it programmatically  
- Go to this [link](https://developer.github.com/v3/guides/getting-started/#create-a-repository) to gain a deeper understanding of this issue 

```bash
AUTH="Authorization: token $GIT_TOKEN"
DATA='{"name":"dotfiles","private": true}' # create a JSON structure
curl -H "$AUTH" https://api.github.com/user/repos -d "$DATA"
```
//slide//

## Git Basics - Working with Remotes

- Final required setup is to tell our local directory where it's remote counterpart is located
- Again to understand this concept further you can (but don't have to) go [here](https://git-scm.com/book/en/v2/Git-Basics-Working-with-Remotes)

```bash
# Tell the folder where the remote repository is
git remote add origin git@github.com:<git_user_name>/dotfiles.git
``` 
//slide//

## Let's add some files to our repository

- You should now have an empty repository on GitHub with a link resembling this `https://github.com/<username>/dotfiles`
- Goto the link (change the user name to your git account) and confirm that it is empty
- We want to tell git which files to upload to the remote folder 
- Let's add files with the `add` command

```bash
git add .bashrc # adding a single file
git add --all # To add everything in your directory
```

//slide//

## Let's commit and push files to the remote repository

- Recall that git is much more than a method to backup files and structure of a project
- The more important aspect of git is tracking the **evolution** of the project 
- Which mean that we can add a file or files multiple times 
- Committing is the process of adding meta information about the current changes made to the repository 


//slide//

## Let's commit and push files to the remote repository


- In practice, we want a short (50 char) sentence describing what changes you made to the content 

```bash
# Make nano your default editor  
echo "export EDITOR='nano'" >> ~/.bashrc
. ~/.bashrc 
# Tell git we want to commit and add why 
git commit -m "updating dotfiles on $(date -u)"
# Push our files to the web
git push -u origin master
``` 

//slide//

## Go to your new repository 

- Use the web interface to change one of your files 
- One option is to add some of the aliases I showed you to the .bash_aliases file
- Here is a  <a href="#/alias">link</a> to the relevant slide
- It is much easier to edit these kinds on the web (or on your local computer)
- Now we will pull the updated repository back to the cluster 

```bash
cd ~/dotfiles
git status
git pull origin master
``` 

- You should now have an updated .bash_aliases on your cluster 


//slide//

## Final step for this week 

- Please send me an email with the email you have on your github account 
- This should be addressed to eyal.soreq@mpi-mail.mpg.de 
- The topic should be add me as a collaborator to the course github
- See you online 


//slide//

## Congratulations you are the proud owner of a `dotfiles` repository

- This was a cool but annoying process 
- Next we will cover how to collaborate to a repository you don't own 
- After that we will revisit other important git commands and end this weeks material
- Take a well earned break and move forward when ready



