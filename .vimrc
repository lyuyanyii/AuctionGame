set nocompatible
color evening
set bs=2
set number
set tabstop=4
set softtabstop=4
set noswapfile
set shiftwidth=4
"set smarttab
set autoindent
set smartindent
set cindent
set ruler
set go=
set nobackup
set noswapfile
"set wrap
set guifont=Courier\ New\ 14
"set noshowmatch
set fileencodings=utf-8,gbk
set encoding=utf8
set autochdir
set autoread
set nohlsearch
filetype indent plugin on
set expandtab
syntax on
let mapleader="-"
noremap <leader>q <esc>:%s/    /    /g<CR>
set keywordprg=sdcv
noremap <leader>begin<CR> i#include<cstdio><CR>#include<cstdlib><CR>#include<cstring><CR>#include<algorithm><CR>#include<iostream><CR>#include<fstream><CR>#include<map><CR>#include<ctime><CR>#include<set><CR>#include<queue><CR>#include<cmath><CR>#include<vector><CR>#include<bitset><CR>#include<functional><CR>#define x first<CR>#define y second<CR>#define mp make_pair<CR>#define pb push_back<CR>using namespace std;<CR><CR>typedef long long LL;<CR>typedef double ld;<CR><CR>int main()<CR>{<CR>#ifndef ONLINE_JUDGE<CR><esc>0i    freopen("input.txt","r",stdin);freopen("output.txt","w",stdout);<CR>#endif<CR><esc>0i    return 0;<CR>}<esc>

nnoremap <leader>ev :tabnew $MYVIMRC<cr>
nnoremap <leader>sv :source $MYVIMRC<cr>!
nnoremap <leader>zs :<esc>0i//<esc>
nnoremap <leader>as 1GVG"+y<esc>k<cr>

syntax on
nnoremap <leader>q :!g++ % -o %< <cr>:!./%< <cr>
nnoremap qq :!python3 %<cr>
nnoremap tl :TlistOpen<cr>
nnoremap 11 :w!<cr>:!g++ % -o %< -Wall -O2<cr>:!./%< <cr>
nnoremap 22 :w!<cr>:!xelatex %<cr>:!open %<.pdf<cr>

filetype off
set rtp+=~/.vim/bundle/vundle/
call vundle#begin()
Plugin 'gmarik/Vundle.vim'
Plugin 'tmhedberg/SimpylFold'
Plugin 'vim-scripts/indentpython.vim'
"Plugin 'scrooloose/syntastic'
Bundle 'Valloric/YouCompleteMe'
Plugin 'nvie/vim-flake8'
Plugin 'scrooloose/nerdtree'
Plugin 'jistr/vim-nerdtree-tabs'
Plugin 'kien/ctrlp.vim'
Plugin 'tpope/vim-fugitive'
Plugin 'Lokaltog/powerline', {'rtp': 'powerline/bindings/vim/'}
call vundle#end()
"set clipboard=unnamed

let NERDTreeIgnore=['\.pyc$', '\~$'] "ignore files in NERDTree

let g:ycm_autoclose_preview_window_after_completion=1
let g:ycm_server_python_interpreter='/usr/local/bin/python3'
map <leader>g  :YcmCompleter GoToDefinitionElseDeclaration<CR>

set splitbelow
set splitright
syntax on
filetype on
