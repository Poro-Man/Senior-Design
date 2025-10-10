from importlib.metadata import version
import tiktoken

print("tiktoken version:", version("tiktoken"))

enc = tiktoken.get_encoding("gpt2")
text = "Hello, do you like tea? <|endoftext|> In the sunlit terraces of someunknownPlace."
ids = enc.encode(text, allowed_special={"<|endoftext|>"})
print(ids)




strings = enc.decode(ids)
print(strings)

with open ("the-verdict.txt", "r", encoding ="utf-8") as f:
    raw_text = f.read()
enc_text = enc.encode(raw_text)
print(len(enc_text))

enc_test = enc_text[5:]

context_size = 10 #A
x = enc_test[:context_size]
y = enc_test[1:context_size+1]
print(f"x:  {x}")
print (f"y:     {y}")


for i in range(1, context_size+1):
    context = enc_test[:i]
    desired = enc_test[i]
    print(context, "------>",  desired)

for i in range(1, context_size+1):
    context = enc_test[:i]
    desired = enc_test[i]
    print(enc.decode(context), "---->", enc.decode([desired]))


