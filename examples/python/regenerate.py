# Generates bindings for the ggml library.
#
# cffi requires prior C preprocessing of the headers, and it uses pycparser which chokes on a couple of things
# so we help it a bit (e.g. replace sizeof expressions with their value, remove exotic syntax found in Darwin headers).
import os
import sys
import re
import subprocess
import cffi
from stubs import generate_stubs

API = os.environ.get("API", "api.h")
CC = os.environ.get("CC") or "gcc"
C_INCLUDE_DIR = os.environ.get("C_INCLUDE_DIR", "../../../llama.cpp")
CPPFLAGS = [
    "-I",
    C_INCLUDE_DIR,
    "-D__fp16=uint16_t",  # pycparser doesn't support __fp16
    "-D__attribute__(x)=",
    "-D_Static_assert(x, m)=",
] + [x for x in os.environ.get("CPPFLAGS", "").split(" ") if x != ""]

try:
    cmd_args = [CC, "-E", *CPPFLAGS, API]
    print(" ".join(cmd_args))
    header = subprocess.run(cmd_args, capture_output=True, text=True, check=True).stdout
except subprocess.CalledProcessError as e:
    print(f"{e.stderr}\n{e}", file=sys.stderr)
    raise

header = "\n".join(
    [l for l in header.split("\n") if not l.startswith("#") and l.strip() != ""]
)
header = ";".join(
    [
        expr
        for expr in header.split(";")
        if not any(
            filtered_str in expr
            # targeted known instances to filter:
            # __darwin_va_list
            # __gnuc_va_list
            for filtered_str in ["_va_list", "__asm__"]
        )
        and not expr.strip().startswith("extern ")
    ]
)  # pycparser hates this
header = "\n".join([l.replace("__restrict ", " ") for l in header.split("\n")])

# Replace constant size expressions w/ their value (compile & run a mini exe for each, because why not).
# First, extract anyting *inside* square brackets and anything that looks like a sizeof call.
for expr in set(re.findall(f"(?<=\\[)[^\\]]+(?=])|sizeof\\s*\\([^()]+\\)", header)):
    if re.match(r"^(\d+|\s*)$", expr):
        continue  # skip constants and empty bracket contents
    subprocess.run(
        [CC, "-o", "eval_size_expr", *CPPFLAGS, "-x", "c", "-"],
        text=True,
        check=True,
        input=f"""#include <stdio.h>
                             #include "{API}"
                             int main() {{ printf("%lu", (size_t)({expr})); }}""",
    )
    size = subprocess.run(
        ["./eval_size_expr"], capture_output=True, text=True, check=True
    ).stdout
    print(f"Computed constexpr {expr} = {size}")
    header = header.replace(expr, size)

ffibuilder = cffi.FFI()
ffibuilder.cdef(header)
ffibuilder.set_source(
    f"ggml.cffi", None
)  # we're not compiling a native extension, as this quickly gets hairy
ffibuilder.compile(verbose=True)

with open("ggml/__init__.pyi", "wt") as f:
    f.write(generate_stubs(header))
