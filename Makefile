PYTHON := python3
PIP := pip3
VENV_DIR := venv
SRC_DIR := .
REQ_FILE := requirements.txt
TARGET1 := Rede_Neural.py
TARGET2 := Regressao_Linear.py

.PHONY: all run run1 run2 install-deps clean venv

# Tarefa padrão: executa ambos os scripts
all: run

# Criar ambiente virtual
venv:
	$(PYTHON) -m venv $(VENV_DIR)

# Instalar dependências no ambiente virtual
install-deps: venv
	$(VENV_DIR)/bin/$(PIP) install -r $(REQ_FILE)

# Executar o primeiro programa dentro do ambiente virtual
run1: install-deps
	$(VENV_DIR)/bin/$(PYTHON) $(SRC_DIR)/$(TARGET1)

# Executar o segundo programa dentro do ambiente virtual
run2: install-deps
	$(VENV_DIR)/bin/$(PYTHON) $(SRC_DIR)/$(TARGET2)

# Executar ambos os programas sequencialmente
run: run1 run2

# Limpar ambiente virtual e arquivos temporários
clean:
	rm -rf $(VENV_DIR)
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
