import json
import random

# Lista completa de produtos financeiros
produtos_financeiros = [
    "Cartao de Credito", # Quantidade
    "Conta Corrente",    # Valor
    "Poupanca",          # Valor
    "CDB",               # Valor
    "LCI",               # Valor
    "LCA",               # Valor
    "Credito Pessoal",   # Valor
    "FII",               # Valor
    "Acoes",             # Valor
    "ETF",               # Valor
    "Fundos de Investimento Multimercado", # Valor
    "Fundos de Investimento de Renda Fixa", # Valor
    "Fundos de Investimento de Acoes",    # Valor
    "PGBL",              # Valor
    "VGBL",              # Valor
    "Credito Imobiliario", # Valor
    "Consorcio",         # Valor
    "Fundo de Capitalizacao" # Valor
]

# Produtos que terão valor de quantidade (apenas Cartao de Credito)
produtos_quantidade = ["Cartao de Credito"]

# Listas de nomes para simular nomes reais
nomes_primeiro = ["Ana", "João", "Maria", "Pedro", "Fernanda", "Carlos", "Juliana", "Roberto", "Gabriela", "Fernando", "Beatriz", "Antônio", "Camila", "Paulo", "Larissa", "Ricardo", "Amanda", "Thiago", "Patrícia", "Rafael"]
nomes_ultimo = ["Silva", "Santos", "Oliveira", "Souza", "Costa", "Pereira", "Almeida", "Lima", "Rodrigues", "Martins", "Fernandes", "Gomes", "Lopes", "Mendes", "Nascimento", "Carvalho", "Araújo", "Moraes", "Castro", "Dias"]

def gerar_nome_aleatorio():
    primeiro = random.choice(nomes_primeiro)
    ultimo = random.choice(nomes_ultimo)
    return f"{primeiro} {ultimo}"

def gerar_valor_produto(produto):
    if produto in produtos_quantidade:
        # Quantidade para Cartao de Credito (1 a 5 cartões)
        return random.randint(1, 5)
    else:
        # Valores monetários simulados para outros produtos
        if produto in ["Conta Corrente", "Poupanca"]:
            return random.randint(100, 100000) # Saldos menores/médios
        elif produto in ["CDB", "LCI", "LCA", "Fundo de Capitalizacao"]:
            return random.randint(5000, 500000) # Investimentos renda fixa / Capitalizacao
        elif produto in ["FII", "Acoes", "ETF", "Fundos de Investimento Multimercado", "Fundos de Investimento de Renda Fixa", "Fundos de Investimento de Acoes"]:
             return random.randint(10000, 2000000) # Investimentos variáveis / fundos
        elif produto in ["PGBL", "VGBL"]:
             return random.randint(50000, 5000000) # Previdência
        elif produto == "Credito Pessoal":
             return random.randint(1000, 100000) # Limite ou saldo devedor
        elif produto == "Credito Imobiliario":
             return random.randint(100000, 3000000) # Saldo devedor ou valor financiado
        elif produto == "Consorcio":
             return random.randint(10000, 1000000) # Saldo a pagar ou bem

        return random.randint(1000, 1000000) # Valor padrão para outros casos não mapeados especificamente

# Gerar a lista de 100 clientes
clientes = []
produtos_outros = [p for p in produtos_financeiros if p not in ["Conta Corrente", "Poupanca"]] # Produtos que podem ser adicionados aleatoriamente

for i in range(1, 101):
    cliente = {}
    cliente["id"] = i
    cliente["idade"] = random.randint(18, 75)
    cliente["nome"] = gerar_nome_aleatorio()

    # Garantir Conta Corrente e Poupança
    cliente["Conta Corrente"] = gerar_valor_produto("Conta Corrente")
    cliente["Poupanca"] = gerar_valor_produto("Poupanca")

    # Adicionar outros produtos aleatoriamente
    num_outros_produtos = random.randint(4, 10) # Cada cliente terá entre 4 e 10 produtos adicionais
    produtos_para_adicionar = random.sample(produtos_outros, min(num_outros_produtos, len(produtos_outros)))

    for produto in produtos_para_adicionar:
        cliente[produto] = gerar_valor_produto(produto)

    clientes.append(cliente)

# Converter a lista de clientes para string JSON formatada
json_output = json.dumps(clientes, indent=2, ensure_ascii=False)

# Para imprimir o JSON completo, execute este script Python
# print(json_output)

# Como o output pode ser muito longo, mostrarei apenas os primeiros 10 e o último cliente como exemplo:
print("[")
for j in range(min(10, len(clientes))):
    print(json.dumps(clientes[j], indent=2, ensure_ascii=False) + ("," if j < len(clientes)-1 else ""))

if len(clientes) > 10:
    print("  ...") # Indicador de que há mais clientes no meio
    print(json.dumps(clientes[-1], indent=2, ensure_ascii=False))

print("]")


# Salvar o JSON em um arquivo
with open("clientes.json", "w", encoding="utf-8") as f:
    f.write(json_output)    