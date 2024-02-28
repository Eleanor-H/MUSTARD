prelude
import init.data.nat.basic init.data.string.basic

def lean.version : nat × nat × nat :=
(3, 51, 1)

def lean.githash : string :=
"cce7990ea86a78bdb383e38ed7f9b5ba93c60ce0"

def lean.is_release : bool :=
1 ≠ 0

/-- Additional version description like "nightly-2018-03-11" -/
def lean.special_version_desc : string :=
""
