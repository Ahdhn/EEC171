/* A Bison parser, made by GNU Bison 2.7.  */

/* Bison interface for Yacc-like parsers in C
   
      Copyright (C) 1984, 1989-1990, 2000-2012 Free Software Foundation, Inc.
   
   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.
   
   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.
   
   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.  */

/* As a special exception, you may create a larger work that contains
   part or all of the Bison parser skeleton and distribute that work
   under terms of your choice, so long as that work isn't itself a
   parser generator using the skeleton or a modified version thereof
   as a parser skeleton.  Alternatively, if you modify or redistribute
   the parser skeleton itself, you may (at your option) remove this
   special exception, which will cause the skeleton and the resulting
   Bison output files to be licensed under the GNU General Public
   License without this special exception.
   
   This special exception was added by the Free Software Foundation in
   version 2.2 of Bison.  */

#ifndef YY_YY_Y_TAB_H_INCLUDED
# define YY_YY_Y_TAB_H_INCLUDED
/* Enabling traces.  */
#ifndef YYDEBUG
# define YYDEBUG 0
#endif
#if YYDEBUG
extern int yydebug;
#endif

/* Tokens.  */
#ifndef YYTOKENTYPE
# define YYTOKENTYPE
   /* Put the tokens into the symbol table, so that GDB and other debuggers
      know about them.  */
   enum yytokentype {
     INT = 258,
     NAME = 259,
     OREQ = 260,
     ANDEQ = 261,
     RSHIFTEQ = 262,
     LSHIFTEQ = 263,
     DIVEQ = 264,
     MULTEQ = 265,
     MINUSEQ = 266,
     PLUSEQ = 267,
     OROR = 268,
     ANDAND = 269,
     NE = 270,
     EQ = 271,
     GE = 272,
     LE = 273,
     RSHIFT = 274,
     LSHIFT = 275,
     UNARY = 276,
     END = 277,
     ALIGN_K = 278,
     BLOCK = 279,
     QUAD = 280,
     LONG = 281,
     SHORT = 282,
     BYTE = 283,
     SECTIONS = 284,
     SIZEOF_HEADERS = 285,
     OUTPUT_FORMAT = 286,
     FORCE_COMMON_ALLOCATION = 287,
     OUTPUT_ARCH = 288,
     INCLUDE = 289,
     MEMORY = 290,
     DEFSYMEND = 291,
     NOLOAD = 292,
     DSECT = 293,
     COPY = 294,
     INFO = 295,
     OVERLAY = 296,
     DEFINED = 297,
     TARGET_K = 298,
     SEARCH_DIR = 299,
     MAP = 300,
     ENTRY = 301,
     SIZEOF = 302,
     NEXT = 303,
     ADDR = 304,
     STARTUP = 305,
     HLL = 306,
     SYSLIB = 307,
     FLOAT = 308,
     NOFLOAT = 309,
     ORIGIN = 310,
     FILL = 311,
     LENGTH = 312,
     CREATE_OBJECT_SYMBOLS = 313,
     INPUT = 314,
     GROUP = 315,
     OUTPUT = 316,
     CONSTRUCTORS = 317,
     ALIGNMOD = 318,
     AT = 319,
     PROVIDE = 320,
     CHIP = 321,
     LIST = 322,
     SECT = 323,
     ABSOLUTE = 324,
     LOAD = 325,
     NEWLINE = 326,
     ENDWORD = 327,
     ORDER = 328,
     NAMEWORD = 329,
     FORMAT = 330,
     PUBLIC = 331,
     BASE = 332,
     ALIAS = 333,
     TRUNCATE = 334,
     REL = 335,
     INPUT_SCRIPT = 336,
     INPUT_MRI_SCRIPT = 337,
     INPUT_DEFSYM = 338
   };
#endif
/* Tokens.  */
#define INT 258
#define NAME 259
#define OREQ 260
#define ANDEQ 261
#define RSHIFTEQ 262
#define LSHIFTEQ 263
#define DIVEQ 264
#define MULTEQ 265
#define MINUSEQ 266
#define PLUSEQ 267
#define OROR 268
#define ANDAND 269
#define NE 270
#define EQ 271
#define GE 272
#define LE 273
#define RSHIFT 274
#define LSHIFT 275
#define UNARY 276
#define END 277
#define ALIGN_K 278
#define BLOCK 279
#define QUAD 280
#define LONG 281
#define SHORT 282
#define BYTE 283
#define SECTIONS 284
#define SIZEOF_HEADERS 285
#define OUTPUT_FORMAT 286
#define FORCE_COMMON_ALLOCATION 287
#define OUTPUT_ARCH 288
#define INCLUDE 289
#define MEMORY 290
#define DEFSYMEND 291
#define NOLOAD 292
#define DSECT 293
#define COPY 294
#define INFO 295
#define OVERLAY 296
#define DEFINED 297
#define TARGET_K 298
#define SEARCH_DIR 299
#define MAP 300
#define ENTRY 301
#define SIZEOF 302
#define NEXT 303
#define ADDR 304
#define STARTUP 305
#define HLL 306
#define SYSLIB 307
#define FLOAT 308
#define NOFLOAT 309
#define ORIGIN 310
#define FILL 311
#define LENGTH 312
#define CREATE_OBJECT_SYMBOLS 313
#define INPUT 314
#define GROUP 315
#define OUTPUT 316
#define CONSTRUCTORS 317
#define ALIGNMOD 318
#define AT 319
#define PROVIDE 320
#define CHIP 321
#define LIST 322
#define SECT 323
#define ABSOLUTE 324
#define LOAD 325
#define NEWLINE 326
#define ENDWORD 327
#define ORDER 328
#define NAMEWORD 329
#define FORMAT 330
#define PUBLIC 331
#define BASE 332
#define ALIAS 333
#define TRUNCATE 334
#define REL 335
#define INPUT_SCRIPT 336
#define INPUT_MRI_SCRIPT 337
#define INPUT_DEFSYM 338



#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED
typedef union YYSTYPE
{
/* Line 2058 of yacc.c  */
#line 62 "./ldgram.y"

  bfd_vma integer;
  char *name;
  int token;
  union etree_union *etree;


/* Line 2058 of yacc.c  */
#line 231 "y.tab.h"
} YYSTYPE;
# define YYSTYPE_IS_TRIVIAL 1
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
#endif

extern YYSTYPE yylval;

#ifdef YYPARSE_PARAM
#if defined __STDC__ || defined __cplusplus
int yyparse (void *YYPARSE_PARAM);
#else
int yyparse ();
#endif
#else /* ! YYPARSE_PARAM */
#if defined __STDC__ || defined __cplusplus
int yyparse (void);
#else
int yyparse ();
#endif
#endif /* ! YYPARSE_PARAM */

#endif /* !YY_YY_Y_TAB_H_INCLUDED  */
