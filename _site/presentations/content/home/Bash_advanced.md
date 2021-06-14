# Some Advanced Bash

- In this section we will cover how to create custom commands (aliases), allowing you to create shortcuts for a single command or a group of commands.
- We will also learn how to configure your personal home account on the cluster 
- How to log on the cluster using the terminal
- And how to transfer files from and to the cluster using rsync


---

<!-- .slide: id="alias" -->

## alias 

- A shell alias is a shortcut to reference a command. It can be used to avoid typing long commands or as a means to correct incorrect input. For common patterns it can reduce keystrokes and improve efficiency. 
- Think about complex commands that you use a lot, or even simple commands with common used options. 
- The format is simple and I will go over some of my favorite (Copy the ones you like)  


---

## alias - examples 

```bash
alias # list all current aliases
alias rm='rm -i' # Will ask permission before deleting files.
alias cp='cp -i' # Will ask permission before overwriting files.
alias mv='mv -i' # Will ask permission before overwriting files.
alias cpv='rsync -ah --info=progress2' # a fancy copy
alias sr='sort -hr' # sort reverse.
```

---

## alias - examples  (cont.)

```bash
alias mkdir='mkdir -p'
alias h='history'
alias hg='history|grep'
alias ..='cd ..'
alias du='du -kh'    
alias dt='du -sh * | sr'
```

---

## alias - examples  (cont.)

```bash
alias ls='ls -h --color'
alias ll="ls -lv --group-directories-first"
alias lm='ll | more'       #  Pipe through 'more'
alias lr='ll -R'           #  Recursive ls.
alias la='ll -A'           #  Show hidden files.
alias lx='ll -XB'          #  Sort by extension.
alias lk='ll -Sr  | sr'    #  Sort by size, biggest first.
alias lt='ll -tr  | sr'    #  Sort by date, most recent first.
```


---

## .dotfile

- If you followed so far you should be able to do the following: 

```bash
cd ~ # go to home dir
ls # list files 
la # use our new flashy alias to list files 
```

- The two list should be different, with the latter containing dotfiles, that are normally hidden.
- Dotfiles are used by different programs to store important information that dramatically changes your system (therefore handle with care). 


---

## .profile 

- A *.profile* file stores user specific environment and startup programs configurations and is loaded each time a shell is created. 
- In the GWDG HPC you don't have a .profile file by default. 
- Let's create it and add into it some crucial code. 

```bash
nano ~/.profile 
# will create the empty .profile and open a text editor
```

---

## .profile 

- This should look like this 

![](/img/Tutorial_nano.png)

---

## .profile 

- Copy and paste this piece of code and close and save using CTRL+X

```bash
if [ -f /etc/bashrc ]; then
    . /etc/bashrc # --> source global /etc/bashrc, if present.
fi
if [ -f . ~/.bashrc ]; then
    . ~/.bashrc # --> source local ~/.bashrc, if present.
fi    
```

---

## .bashrc 

- What we just did was to tell the cluster to source (i.e. load into memory) both global (/etc/bashrc) and local .bashrc files every time a bash shell is created. 

- Lets open our local .bashrc file and add some functionality to it. 

```bash
nano ~/.bashrc 
```

---

## .bashrc 

- So now we are going to create a conditional statement that will source some files to memory
- The conditional statement below will only run if the file exists 
- Using a dot before a file is shorthand to source command

```bash
export PATH=$HOME/bin:$PATH # Add local bin to $PATH 
if [ -f ~/.bash_aliases ]; then
    . ~/.bash_aliases   # --> source local aliases if present.
fi
```

---

## .bash_aliases 

- We already know what aliases are. 
- However, it is important to understand that the aliases you created will only exist as long as the shell is open. 
- For them to be present any time you log on to the cluster we need to source them every time. 
- The code we just wrote will source the file .bash_aliases if it exists whenever you open a shell. 
- So let's create this file and append two aliases
- Feel free to use nano and copy over as many aliases if you like

```bash
touch ~/.bash_aliases  # create the file 
echo "alias ls='ls -h --color'" >> ~/.bash_aliases 
echo "alias hg='history|grep'" >> ~/.bash_aliases 
```

---

## ssh

- To verify that this works we need to exit the terminal or alternatively open a new one. 
- We could just use the web interface to spawn a new terminal 
- Alternatively we could use Secure Shell (ssh) to log on to the cluster from our personal computer 
- This is important as some times we would like to copy data to and from the cluster and the simplest way to achieve that is by using a terminal. 
- Go over to [CONNECT WITH SSH](https://info.gwdg.de/docs/doku.php?id=en:services:application_services:high_performance_computing:connect_with_ssh) and follow the instructions there.


---
<!-- .slide: id="ssh_login" -->

## ssh Key

- To simplify connectivity between computers we use an Rivest–Shamir–Adleman (rsa) key which creates a two key system to allow secure data transmission (and avoid needing a password)

```bash
# On your personal computer generate your rsa 
ssh-keygen -t rsa 
# Copy it over to the cluster 
ssh-copy-id <user_id>@gwdu101.gwdg.de 
# Check by login in 
ssh  <user_id>@gwdu101.gwdg.de
# And log out 
logout 
```


---

<!-- .slide: id="rsync" -->

## rsync

- Rsync is a fast and extraordinarily versatile file copying tool.
- We are going to learn how to use it with the following steps:
    - Copy to your personal computer your .bash_aliases file 
    - Edit it locally (using any software you like)
    - Transfer it back and overwrite the previous copy 


---

## rsync - pull a file from the remote server

- Lets copy our file from the cluster - this is called pulling a file 
- Open a terminal in your personal computer, create and goto your own sandbox folder

```bash
mkdir ~/sandbox
cd ~/sandbox
rsync <user_id>@gwdu101.gwdg.de:.bash_aliases .bash_aliases
cat .bash_aliases
```

---

## rsync - push a file to the remote server 

- Add a new alias to your file 
- Then push the file to your home directory and overwrite to old one 
- The push syntax is identical to the pull with the source and destination flipped
- We will confirm by passing a command to the cluster using ssh 

```bash
echo "alias la='ll -A'" >> .bash_aliases 
rsync .bash_aliases <user_id>@gwdu101.gwdg.de:.bash_aliases 
ssh  esoreq@gwdu101.gwdg.de "cat ~/.bash_aliases"
```

---

## Summary 


- In this section we covered more advanced topics 
    - We learned to log on to the cluster using ssh 
    - We created aliases to simplify our workflow 
    - And learned to copy from and to the cluster 
- As you progress in your understanding you will find that interacting with the cluster is most efficient via the terminal.
- However, in terms of reproducibility (and especially for beginners) it is better to use the Jupyter framework we will cover shortly  
- Time to take a short break and move to setting up version control 





