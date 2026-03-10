mkdir -p ~/.streamlit/

echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
" > ~/.streamlit/config.toml

# Force Python 3.9
pyenv install -s 3.9.18
pyenv global 3.9.18
