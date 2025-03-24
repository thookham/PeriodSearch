#pragma once

#include "arrayHelpers.hpp"

#ifndef LCHELPERS_H
#define LCHELPERS_H

int PrepareLcData(globalsCl& gl, const char* filename);
//globals PrepareLcData(const char* filename);

void MakeConvexityRegularization(struct globalsCl& gl);

#endif
