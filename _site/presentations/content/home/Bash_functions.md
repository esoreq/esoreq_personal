
# Even More Advanced Bash

- In this section we will go over some more advanced Bash concepts 
- We will also cover how to create functions to automate things
- I will share with you some functions I created to simplify my life in the gwdg cluster
- If you want to expand your Bash skills [this](https://www.tldp.org/LDP/abs/html/abs-guide.html) is a good place to start 


//slide//

## Conditionals 

- We previously used the `IF` statement in our `.bashrc` file. 
- The most basic form of conditionals is: 
```bash
if [condition is true] then [do something]. 
```
- We can extend this by adding `else` 

```bash
if [condition is true] 
then 
    [do something] 
else [condition is false] 
then 
    [do something else ]. 
```

//slide//

## Conditionals 

- Using Bash syntax (i.e. the set of rules that defines the combinations of symbols that are considered to be a correctly structured and therefore usable code)
- Our previous example will look like this: 

```bash
if` [expression]; then 
    [do something] 
else 
    [do something else] 
fi
```

//slide//

## Conditionals 

- This can be extended even further using else if which allows us to add conditions and in fact create hierarchy.  
- Consider the following program 
- Try to understand this and goto the next slide for a breakdown

```bash 
cTime=$(date +%H)
if [ $cTime -gt 06 -a $cTime -le 12 ]; then
    echo "Good Morning"
elif [ $cTime -gt 12 -a $cTime -le 18 ]; then
    echo "Good Afternoon"
else
    echo "Good night"
fi
```

//slide//

## Let's unpack 

| | |
| :-- | :-- |
| `cTime=$(date +%H)` | store the current hour in a variable |
| `IF` | Start the conditional statement |
| `$cTime -gt 06` | Check if time is greater than 6 |
| `-a` | logical AND |
| `$cTime -le 12` | less or equal to 12 |
| `if true` | `echo "Good Morning"` |
| `ELSE IF` | Another conditional statement  | 
| `cTime > 12 & cTime <= 18` | Same as above |      
| `if true` | `echo "Good Afternoon"` |
| `ELSE` | `i.e. cTime > 18 & cTime <= 6` |
| `if both are false` | `echo "Good Afternoon"` |
| | |


//slide//

## Making this into a function is dead simple 

```bash 
function greeting {
cTime=$(date +%H)
if [ $cTime -gt 06 -a $cTime -le 12 ]; then
    echo "Good Morning"
elif [ $cTime -gt 12 -a $cTime -le 18 ]; then
    echo "Good Afternoon"
else
    echo "Good night"
fi;
}
```

//slide//

## We can also add variables  

- The dollar $ sign with a number will capture variables that our function can use  


```bash 
function greeting {
cTime=$(date +%H)
if [ $cTime -gt 06 -a $cTime -le 12 ]; then
    echo "Good Morning $1"
elif [ $cTime -gt 12 -a $cTime -le 18 ]; then
    echo "Good Afternoon $1"
else
    echo "Good night $1"
fi;
}
greeting "Your Name" # will add a name as a variable 
```


//slide//

## .Bash_functions

- Like the aliases we want to simplify our life (and reduce typing) by creating functions that take anything that we will use often and convert it into a function. 
- And we obviously want these functions to be there anytime we log into the cluster
- Let's create a .Bash_functions file and make it load on login
- Any useful function should go in this file 
- Try to do this on your own and go to the next slide and see my example


//slide//

## .Bash_functions


```bash
touch ~/.Bash_functions # create a dot functions file 
# now add this to ~/.bashrc file to source when you login
if [ -f ~/.bash_functions ]; then
    . ~/.bash_functions   # --> source local functions if present.
fi
```

//slide//

## No let's go over some of my most useful functions

- Function mcd will create a directory and then immediately move into that directory. 

```bash
# Create a directory and move into it
function  mcd {
    mkdir -p $1
    cd $1
}
```

//slide//

## function sp - step up 

- This one will step up in the directory hierarchy the specified number of steps
- Again try to see what and how I do things here

```bash
# Step up the folder tree $1 times  
function sp {
if [ -z "$1" ]; then
    echo "You forgot to give a number!"
elif [ $1 -eq $1 ]; then
    if [ $1 -gt 9 -o $1 -lt 1 ]; then
        echo "$1 is not a valid variable"
    else
        for i in $(seq 1 $1); do
            cd ..
        done;
    fi;
fi;
}
```

//slide//

## function pull - simplify git pull 

```bash
# will go to a git folder pull an updated version and return to origin
function pull {
  if [ ! -z "$1" ]; then
    origin=$PWD
    cd $1  
    git pull origin master
    cd $origin
  else
    git pull origin master
  fi
}
```

//slide//

## function push - simplify git push 

```bash
# will go to a git folder push a committed folder and return to origin
function push {
  if [ ! -z "$1" ]; then
    origin=$PWD
    cd $1  
    git push origin master
    cd $origin
  else
    git push origin master
  fi
}
```

//slide//

## function commit - simplify git commit 

```bash
# will go to a git folder 
# Add all files, commit them and push to remote  
function commit {
  if [ ! -z "$1" ]; then
    origin=$PWD
    cd $1
    git add --all 
    if [ ! -z "$2"] ; then
        git commit -m $2
    else         
        git commit -m "routine update on $(date -u)" 
    fi    
    git push origin master
    cd $origin
  else
    echo "This functions requires some input"
  fi
}
```

//slide//

## function remote - simplify git remote 

```bash
# Create a new repository on GitHub 
# Syntax newRepo $repo_name $GIT_USER $GIT_TOKEN
function remote {
  if [ ! -z "$1" -a ! -z "$2" ]; then
    REPO=git@github.com:$1/$2.git
    git remote add origin $REPO
  else
    echo "Give username and repository name"
  fi
}
```
//slide//

## function newRepo - simplify new git repository creation 

```bash
# Create a new repository on GitHub 
# Syntax newRepo $repo_name $GIT_USER $GIT_TOKEN
function newRepo {
  if [ -z $1 -a -z $2 -a -z $3 ]; then
    echo "You need to give your repository a name"
  else
    mcd $1   # create and goto the folder
    git init # make a git initialized folder 
    AUTH="Authorization: token $3"
    DATA=$(printf '{"name":"%s"}' $1)
    API="https://api.github.com/user/repos"
    curl -H "$AUTH" https://api.github.com/user/repos -d "$DATA"
    remote $2 $1 
  fi
}
```

//slide//

## function delRepo - simplify git repository deletion 

```bash
# Delete a repository on GitHub 
# Syntax delRepo $repo_name $GIT_TOKEN $GIT_USER  
function delRepo {
  if [ -z $1 -a -z $2 -a -z $3 ]; then
    echo "You need to give your repository a name"
  else
    AUTH="Authorization: token $2"
    curl -X DELETE -H "$AUTH" https://api.github.com/repos/$3/$1 
    rm -rf ~/github/$1
  fi
}
```
//slide//

## function clone - simplify git repository cloning 

- A clone is a local copy of some remote repository
- We want our clones to be in one tidy place 
- For me its a folder called github

```bash
# make sure all cloned repositories are stored in one neat place
function clone {
  if [ -z $1 ]; then
    echo "You need to give a git url"
  else
    origin=$PWD
    if [ -z $2 ]; then
        cd ~/github
    else 
        cd $2
    fi
    git clone $1
    cd $origin
  fi
}
```

//slide//

## Can anybody push files to my repository on github?

- Nobody can push directly to your repository if you are not already granting them write access
- This is true even if your repository isn't private 
- A private repository simply means that only approved users can see on-line content 
- The process of contributing to a public repository in GitHub starts by forking the repository
- For those interested [here is nice summary](https://blog.scottlowe.org/2015/01/27/using-fork-branch-git-workflow/)
- In this course we will not require such process instead we will use a shared space on the cluster. 
 


//slide//

## Summary 

- Ok this almost sums up your introduction to bash and the cluster 
- During the course we will probably add a few more aliases and functions
- This is a good opportunity to go over the various commands and see that you understood everything 
- Also on a personal note, please make note for anything that isn't clear so I can learn from your experience
- When you are ready press on the right arrow for the final section and your home assignment 




