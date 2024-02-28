/*
Copyright (c) 2016 Microsoft Corporation. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.

Author: Leonardo de Moura
*/
#pragma once

namespace lean {
void initialize_vm_core_module();
void finalize_vm_core_module();

void initialize_vm_module();
void finalize_vm_module();
}
