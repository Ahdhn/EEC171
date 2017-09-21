/* ldmain.h -
   Copyright 1991, 1992, 1993 Free Software Foundation, Inc.

   This file is part of GLD, the Gnu Linker.

   GLD is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 1, or (at your option)
   any later version.

   GLD is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with GLD; see the file COPYING.  If not, write to
   the Free Software Foundation, 675 Mass Ave, Cambridge, MA 02139, USA.  */

#ifndef LDMAIN_H
#define LDMAIN_H

extern char *program_name;
extern bfd *output_bfd;
extern char *default_target;
extern boolean trace_files;
extern boolean trace_file_tries;
extern boolean version_printed;
extern int g_switch_value;
extern const char *output_filename;
extern struct bfd_link_info link_info;

extern void add_ysym PARAMS ((const char *));
extern void add_keepsyms_file PARAMS ((const char *filename));

#endif
