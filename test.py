import re

text = """
<prog>
  some code here
  line 2
</prog>

<prog>
  another block
</prog>
"""

pattern = re.compile(r'(?s)<prog>.*?</prog>')
matches = pattern.findall(text)

for i, block in enumerate(matches, 1):
    print(f"Block {i}:\n{block}\n")
