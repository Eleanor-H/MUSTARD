/*
Copyright (c) 2016 Microsoft Corporation. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.

Author: Leonardo de Moura
*/
#pragma once
namespace lean {
vm_obj change(expr const & e, tactic_state const & s);
void initialize_change_tactic();
void finalize_change_tactic();
}
