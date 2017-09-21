/* A Bison parser, made by GNU Bison 2.7.  */

/* Bison implementation for Yacc-like parsers in C
   
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

/* C LALR(1) parser skeleton written by Richard Stallman, by
   simplifying the original so-called "semantic" parser.  */

/* All symbols defined below should begin with yy or YY, to avoid
   infringing on user name space.  This should be done even for local
   variables, as they might otherwise be expanded by user macros.
   There are some unavoidable exceptions within include files to
   define necessary library symbols; they are noted "INFRINGES ON
   USER NAME SPACE" below.  */

/* Identify Bison output.  */
#define YYBISON 1

/* Bison version.  */
#define YYBISON_VERSION "2.7"

/* Skeleton name.  */
#define YYSKELETON_NAME "yacc.c"

/* Pure parsers.  */
#define YYPURE 0

/* Push parsers.  */
#define YYPUSH 0

/* Pull parsers.  */
#define YYPULL 1




/* Copy the first part of user declarations.  */
/* Line 371 of yacc.c  */
#line 21 "./ldgram.y"

/*

 */

#define DONTDECLARE_MALLOC

#include "bfd.h"
#include "sysdep.h"
#include "bfdlink.h"
#include "ld.h"    
#include "ldexp.h"
#include "ldver.h"
#include "ldlang.h"
#include "ldemul.h"
#include "ldfile.h"
#include "ldmisc.h"
#include "ldmain.h"
#include "mri.h"
#include "ldlex.h"

static int typebits;

lang_memory_region_type *region;


char *current_file;
boolean ldgram_want_filename = true;
boolean had_script = false;
boolean force_make_executable = false;

boolean ldgram_in_script = false;
boolean ldgram_had_equals = false;


#define ERROR_NAME_MAX 20
static char *error_names[ERROR_NAME_MAX];
static int error_index;
#define PUSH_ERROR(x) if (error_index < ERROR_NAME_MAX) error_names[error_index] = x; error_index++;
#define POP_ERROR()   error_index--;

/* Line 371 of yacc.c  */
#line 110 "y.tab.c"

# ifndef YY_NULL
#  if defined __cplusplus && 201103L <= __cplusplus
#   define YY_NULL nullptr
#  else
#   define YY_NULL 0
#  endif
# endif

/* Enabling verbose error messages.  */
#ifdef YYERROR_VERBOSE
# undef YYERROR_VERBOSE
# define YYERROR_VERBOSE 1
#else
# define YYERROR_VERBOSE 0
#endif

/* In a future release of Bison, this section will be replaced
   by #include "y.tab.h".  */
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
/* Line 387 of yacc.c  */
#line 62 "./ldgram.y"

  bfd_vma integer;
  char *name;
  int token;
  union etree_union *etree;


/* Line 387 of yacc.c  */
#line 327 "y.tab.c"
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

/* Copy the second part of user declarations.  */

/* Line 390 of yacc.c  */
#line 355 "y.tab.c"

#ifdef short
# undef short
#endif

#ifdef YYTYPE_UINT8
typedef YYTYPE_UINT8 yytype_uint8;
#else
typedef unsigned char yytype_uint8;
#endif

#ifdef YYTYPE_INT8
typedef YYTYPE_INT8 yytype_int8;
#elif (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
typedef signed char yytype_int8;
#else
typedef short int yytype_int8;
#endif

#ifdef YYTYPE_UINT16
typedef YYTYPE_UINT16 yytype_uint16;
#else
typedef unsigned short int yytype_uint16;
#endif

#ifdef YYTYPE_INT16
typedef YYTYPE_INT16 yytype_int16;
#else
typedef short int yytype_int16;
#endif

#ifndef YYSIZE_T
# ifdef __SIZE_TYPE__
#  define YYSIZE_T __SIZE_TYPE__
# elif defined size_t
#  define YYSIZE_T size_t
# elif ! defined YYSIZE_T && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
#  include <stddef.h> /* INFRINGES ON USER NAME SPACE */
#  define YYSIZE_T size_t
# else
#  define YYSIZE_T unsigned int
# endif
#endif

#define YYSIZE_MAXIMUM ((YYSIZE_T) -1)

#ifndef YY_
# if defined YYENABLE_NLS && YYENABLE_NLS
#  if ENABLE_NLS
#   include <libintl.h> /* INFRINGES ON USER NAME SPACE */
#   define YY_(Msgid) dgettext ("bison-runtime", Msgid)
#  endif
# endif
# ifndef YY_
#  define YY_(Msgid) Msgid
# endif
#endif

/* Suppress unused-variable warnings by "using" E.  */
#if ! defined lint || defined __GNUC__
# define YYUSE(E) ((void) (E))
#else
# define YYUSE(E) /* empty */
#endif

/* Identity function, used to suppress warnings about constant conditions.  */
#ifndef lint
# define YYID(N) (N)
#else
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static int
YYID (int yyi)
#else
static int
YYID (yyi)
    int yyi;
#endif
{
  return yyi;
}
#endif

#if ! defined yyoverflow || YYERROR_VERBOSE

/* The parser invokes alloca or malloc; define the necessary symbols.  */

# ifdef YYSTACK_USE_ALLOCA
#  if YYSTACK_USE_ALLOCA
#   ifdef __GNUC__
#    define YYSTACK_ALLOC __builtin_alloca
#   elif defined __BUILTIN_VA_ARG_INCR
#    include <alloca.h> /* INFRINGES ON USER NAME SPACE */
#   elif defined _AIX
#    define YYSTACK_ALLOC __alloca
#   elif defined _MSC_VER
#    include <malloc.h> /* INFRINGES ON USER NAME SPACE */
#    define alloca _alloca
#   else
#    define YYSTACK_ALLOC alloca
#    if ! defined _ALLOCA_H && ! defined EXIT_SUCCESS && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
#     include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
      /* Use EXIT_SUCCESS as a witness for stdlib.h.  */
#     ifndef EXIT_SUCCESS
#      define EXIT_SUCCESS 0
#     endif
#    endif
#   endif
#  endif
# endif

# ifdef YYSTACK_ALLOC
   /* Pacify GCC's `empty if-body' warning.  */
#  define YYSTACK_FREE(Ptr) do { /* empty */; } while (YYID (0))
#  ifndef YYSTACK_ALLOC_MAXIMUM
    /* The OS might guarantee only one guard page at the bottom of the stack,
       and a page size can be as small as 4096 bytes.  So we cannot safely
       invoke alloca (N) if N exceeds 4096.  Use a slightly smaller number
       to allow for a few compiler-allocated temporary stack slots.  */
#   define YYSTACK_ALLOC_MAXIMUM 4032 /* reasonable circa 2006 */
#  endif
# else
#  define YYSTACK_ALLOC YYMALLOC
#  define YYSTACK_FREE YYFREE
#  ifndef YYSTACK_ALLOC_MAXIMUM
#   define YYSTACK_ALLOC_MAXIMUM YYSIZE_MAXIMUM
#  endif
#  if (defined __cplusplus && ! defined EXIT_SUCCESS \
       && ! ((defined YYMALLOC || defined malloc) \
	     && (defined YYFREE || defined free)))
#   include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
#   ifndef EXIT_SUCCESS
#    define EXIT_SUCCESS 0
#   endif
#  endif
#  ifndef YYMALLOC
#   define YYMALLOC malloc
#   if ! defined malloc && ! defined EXIT_SUCCESS && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
void *malloc (YYSIZE_T); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
#  ifndef YYFREE
#   define YYFREE free
#   if ! defined free && ! defined EXIT_SUCCESS && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
void free (void *); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
# endif
#endif /* ! defined yyoverflow || YYERROR_VERBOSE */


#if (! defined yyoverflow \
     && (! defined __cplusplus \
	 || (defined YYSTYPE_IS_TRIVIAL && YYSTYPE_IS_TRIVIAL)))

/* A type that is properly aligned for any stack member.  */
union yyalloc
{
  yytype_int16 yyss_alloc;
  YYSTYPE yyvs_alloc;
};

/* The size of the maximum gap between one aligned stack and the next.  */
# define YYSTACK_GAP_MAXIMUM (sizeof (union yyalloc) - 1)

/* The size of an array large to enough to hold all stacks, each with
   N elements.  */
# define YYSTACK_BYTES(N) \
     ((N) * (sizeof (yytype_int16) + sizeof (YYSTYPE)) \
      + YYSTACK_GAP_MAXIMUM)

# define YYCOPY_NEEDED 1

/* Relocate STACK from its old location to the new one.  The
   local variables YYSIZE and YYSTACKSIZE give the old and new number of
   elements in the stack, and YYPTR gives the new location of the
   stack.  Advance YYPTR to a properly aligned location for the next
   stack.  */
# define YYSTACK_RELOCATE(Stack_alloc, Stack)				\
    do									\
      {									\
	YYSIZE_T yynewbytes;						\
	YYCOPY (&yyptr->Stack_alloc, Stack, yysize);			\
	Stack = &yyptr->Stack_alloc;					\
	yynewbytes = yystacksize * sizeof (*Stack) + YYSTACK_GAP_MAXIMUM; \
	yyptr += yynewbytes / sizeof (*yyptr);				\
      }									\
    while (YYID (0))

#endif

#if defined YYCOPY_NEEDED && YYCOPY_NEEDED
/* Copy COUNT objects from SRC to DST.  The source and destination do
   not overlap.  */
# ifndef YYCOPY
#  if defined __GNUC__ && 1 < __GNUC__
#   define YYCOPY(Dst, Src, Count) \
      __builtin_memcpy (Dst, Src, (Count) * sizeof (*(Src)))
#  else
#   define YYCOPY(Dst, Src, Count)              \
      do                                        \
        {                                       \
          YYSIZE_T yyi;                         \
          for (yyi = 0; yyi < (Count); yyi++)   \
            (Dst)[yyi] = (Src)[yyi];            \
        }                                       \
      while (YYID (0))
#  endif
# endif
#endif /* !YYCOPY_NEEDED */

/* YYFINAL -- State number of the termination state.  */
#define YYFINAL  11
/* YYLAST -- Last index in YYTABLE.  */
#define YYLAST   775

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  107
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  62
/* YYNRULES -- Number of rules.  */
#define YYNRULES  192
/* YYNRULES -- Number of states.  */
#define YYNSTATES  372

/* YYTRANSLATE(YYLEX) -- Bison symbol number corresponding to YYLEX.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   338

#define YYTRANSLATE(YYX)						\
  ((unsigned int) (YYX) <= YYMAXUTOK ? yytranslate[YYX] : YYUNDEFTOK)

/* YYTRANSLATE[YYLEX] -- Bison symbol number corresponding to YYLEX.  */
static const yytype_uint8 yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,   105,     2,     2,     2,    33,    20,     2,
      36,   102,    31,    29,   100,    30,     2,    32,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,    15,   101,
      23,     5,    24,    14,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,   103,     2,   104,    19,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,    44,    18,    45,   106,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     1,     2,     3,     4,
       6,     7,     8,     9,    10,    11,    12,    13,    16,    17,
      21,    22,    25,    26,    27,    28,    34,    35,    37,    38,
      39,    40,    41,    42,    43,    46,    47,    48,    49,    50,
      51,    52,    53,    54,    55,    56,    57,    58,    59,    60,
      61,    62,    63,    64,    65,    66,    67,    68,    69,    70,
      71,    72,    73,    74,    75,    76,    77,    78,    79,    80,
      81,    82,    83,    84,    85,    86,    87,    88,    89,    90,
      91,    92,    93,    94,    95,    96,    97,    98,    99
};

#if YYDEBUG
/* YYPRHS[YYN] -- Index of the first RHS symbol of rule number YYN in
   YYRHS.  */
static const yytype_uint16 yyprhs[] =
{
       0,     0,     3,     6,     9,    12,    14,    15,    20,    21,
      24,    28,    29,    32,    37,    39,    41,    44,    46,    51,
      56,    60,    63,    68,    72,    77,    82,    87,    90,    93,
      96,   101,   106,   109,   112,   113,   117,   120,   121,   123,
     127,   129,   133,   134,   137,   140,   141,   143,   145,   147,
     149,   151,   153,   155,   157,   162,   167,   172,   177,   186,
     191,   193,   198,   199,   205,   210,   211,   217,   219,   223,
     226,   231,   234,   237,   238,   243,   246,   248,   252,   254,
     255,   260,   261,   267,   268,   274,   277,   279,   281,   283,
     285,   290,   295,   298,   300,   301,   303,   305,   307,   309,
     311,   314,   315,   317,   319,   321,   323,   325,   327,   329,
     331,   333,   335,   339,   343,   350,   352,   353,   359,   362,
     366,   367,   368,   376,   380,   384,   388,   389,   394,   399,
     403,   407,   409,   414,   418,   419,   421,   423,   424,   427,
     430,   434,   439,   442,   445,   448,   452,   456,   460,   464,
     468,   472,   476,   480,   484,   488,   492,   496,   500,   504,
     508,   512,   518,   522,   526,   531,   533,   535,   540,   545,
     550,   555,   560,   562,   567,   568,   569,   570,   571,   572,
     573,   588,   590,   592,   594,   596,   598,   599,   602,   608,
     610,   615,   618
};

/* YYRHS -- A `-1'-separated list of the rules' RHS.  */
static const yytype_int16 yyrhs[] =
{
     108,     0,    -1,    97,   119,    -1,    98,   112,    -1,    99,
     110,    -1,     4,    -1,    -1,   111,     4,     5,   158,    -1,
      -1,   113,   114,    -1,   114,   115,    87,    -1,    -1,    82,
     158,    -1,    82,   158,   100,   158,    -1,     4,    -1,    83,
      -1,    89,   116,    -1,    88,    -1,    92,     4,     5,   158,
      -1,    92,     4,   100,   158,    -1,    92,     4,   158,    -1,
      91,     4,    -1,    84,     4,   100,   158,    -1,    84,     4,
     158,    -1,    84,     4,     5,   158,    -1,    37,     4,     5,
     158,    -1,    79,     4,     5,   158,    -1,    85,   118,    -1,
      86,   117,    -1,    90,     4,    -1,    94,     4,   100,     4,
      -1,    94,     4,   100,     3,    -1,    93,   158,    -1,    95,
       3,    -1,    -1,   116,   100,     4,    -1,   116,     4,    -1,
      -1,     4,    -1,   117,   100,     4,    -1,     4,    -1,   118,
     100,     4,    -1,    -1,   120,   121,    -1,   121,   122,    -1,
      -1,   143,    -1,   126,    -1,   150,    -1,   151,    -1,   153,
      -1,   155,    -1,   128,    -1,   101,    -1,    59,    36,     4,
     102,    -1,    60,    36,   109,   102,    -1,    77,    36,   109,
     102,    -1,    47,    36,     4,   102,    -1,    47,    36,     4,
     100,     4,   100,     4,   102,    -1,    49,    36,     4,   102,
      -1,    48,    -1,    75,    36,   125,   102,    -1,    -1,    76,
     123,    36,   125,   102,    -1,    61,    36,   109,   102,    -1,
      -1,    50,   109,   124,   121,    35,    -1,     4,    -1,   125,
     100,     4,    -1,   125,     4,    -1,    43,    44,   127,    45,
      -1,   127,   160,    -1,   127,   128,    -1,    -1,    62,    36,
       4,   102,    -1,   141,   140,    -1,     4,    -1,   129,   142,
       4,    -1,     4,    -1,    -1,   103,   131,   129,   104,    -1,
      -1,     4,   132,    36,   129,   102,    -1,    -1,    31,   133,
      36,   129,   102,    -1,   141,   140,    -1,    74,    -1,   101,
      -1,    78,    -1,   130,    -1,   137,    36,   158,   102,    -1,
      72,    36,   158,   102,    -1,   135,   134,    -1,   134,    -1,
      -1,   135,    -1,    39,    -1,    40,    -1,    41,    -1,    42,
      -1,     5,   156,    -1,    -1,    13,    -1,    12,    -1,    11,
      -1,    10,    -1,     9,    -1,     8,    -1,     7,    -1,     6,
      -1,   101,    -1,   100,    -1,     4,     5,   156,    -1,     4,
     139,   156,    -1,    81,    36,     4,     5,   156,   102,    -1,
     100,    -1,    -1,    51,    44,   145,   144,    45,    -1,   144,
     145,    -1,   144,   100,   145,    -1,    -1,    -1,     4,   146,
     149,    15,   147,   142,   148,    -1,    71,     5,   156,    -1,
      73,     5,   156,    -1,    36,     4,   102,    -1,    -1,    66,
      36,   109,   102,    -1,    67,    36,   152,   102,    -1,    67,
      36,   102,    -1,   152,   142,   109,    -1,   109,    -1,    68,
      36,   154,   102,    -1,   154,   142,   109,    -1,    -1,    69,
      -1,    70,    -1,    -1,   157,   158,    -1,    30,   158,    -1,
      36,   158,   102,    -1,    64,    36,   158,   102,    -1,   105,
     158,    -1,    29,   158,    -1,   106,   158,    -1,   158,    31,
     158,    -1,   158,    32,   158,    -1,   158,    33,   158,    -1,
     158,    29,   158,    -1,   158,    30,   158,    -1,   158,    28,
     158,    -1,   158,    27,   158,    -1,   158,    22,   158,    -1,
     158,    21,   158,    -1,   158,    26,   158,    -1,   158,    25,
     158,    -1,   158,    23,   158,    -1,   158,    24,   158,    -1,
     158,    20,   158,    -1,   158,    19,   158,    -1,   158,    18,
     158,    -1,   158,    14,   158,    15,   158,    -1,   158,    17,
     158,    -1,   158,    16,   158,    -1,    58,    36,     4,   102,
      -1,     3,    -1,    46,    -1,    63,    36,     4,   102,    -1,
      65,    36,     4,   102,    -1,    85,    36,   158,   102,    -1,
      37,    36,   158,   102,    -1,    38,    36,   158,   102,    -1,
       4,    -1,    80,    36,   158,   102,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,     4,   161,   167,   159,   162,    44,   163,
     136,    45,   164,   168,   138,   165,   142,    -1,    53,    -1,
      54,    -1,    55,    -1,    56,    -1,    57,    -1,    -1,   158,
      15,    -1,   158,    36,   166,   102,    15,    -1,    15,    -1,
      36,   166,   102,    15,    -1,    24,     4,    -1,    -1
};

/* YYRLINE[YYN] -- source line where rule number YYN was defined.  */
static const yytype_uint16 yyrline[] =
{
       0,   116,   116,   117,   118,   122,   126,   126,   135,   135,
     145,   146,   150,   151,   152,   155,   158,   159,   160,   162,
     164,   166,   168,   170,   172,   174,   176,   178,   179,   180,
     182,   184,   186,   188,   190,   194,   195,   196,   200,   202,
     206,   208,   213,   213,   224,   225,   231,   232,   233,   234,
     235,   236,   237,   238,   239,   241,   243,   245,   248,   250,
     252,   254,   256,   255,   259,   262,   261,   266,   269,   272,
     278,   282,   283,   284,   288,   290,   294,   296,   301,   306,
     305,   312,   311,   317,   316,   324,   325,   329,   330,   335,
     336,   341,   352,   353,   356,   358,   362,   364,   366,   368,
     373,   380,   386,   388,   390,   392,   394,   396,   398,   400,
     405,   405,   410,   414,   422,   430,   430,   434,   438,   439,
     440,   445,   444,   450,   456,   465,   469,   474,   479,   480,
     485,   487,   493,   495,   497,   501,   503,   508,   508,   514,
     516,   518,   520,   522,   524,   527,   529,   531,   533,   535,
     537,   539,   541,   543,   545,   547,   549,   551,   553,   555,
     557,   559,   561,   563,   565,   567,   569,   572,   574,   576,
     578,   580,   582,   588,   589,   592,   594,   596,   600,   601,
     592,   610,   611,   612,   613,   614,   615,   620,   621,   622,
     623,   627,   629
};
#endif

#if YYDEBUG || YYERROR_VERBOSE || 0
/* YYTNAME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
   First, the terminals, then, starting at YYNTOKENS, nonterminals.  */
static const char *const yytname[] =
{
  "$end", "error", "$undefined", "INT", "NAME", "'='", "OREQ", "ANDEQ",
  "RSHIFTEQ", "LSHIFTEQ", "DIVEQ", "MULTEQ", "MINUSEQ", "PLUSEQ", "'?'",
  "':'", "OROR", "ANDAND", "'|'", "'^'", "'&'", "NE", "EQ", "'<'", "'>'",
  "GE", "LE", "RSHIFT", "LSHIFT", "'+'", "'-'", "'*'", "'/'", "'%'",
  "UNARY", "END", "'('", "ALIGN_K", "BLOCK", "QUAD", "LONG", "SHORT",
  "BYTE", "SECTIONS", "'{'", "'}'", "SIZEOF_HEADERS", "OUTPUT_FORMAT",
  "FORCE_COMMON_ALLOCATION", "OUTPUT_ARCH", "INCLUDE", "MEMORY",
  "DEFSYMEND", "NOLOAD", "DSECT", "COPY", "INFO", "OVERLAY", "DEFINED",
  "TARGET_K", "SEARCH_DIR", "MAP", "ENTRY", "SIZEOF", "NEXT", "ADDR",
  "STARTUP", "HLL", "SYSLIB", "FLOAT", "NOFLOAT", "ORIGIN", "FILL",
  "LENGTH", "CREATE_OBJECT_SYMBOLS", "INPUT", "GROUP", "OUTPUT",
  "CONSTRUCTORS", "ALIGNMOD", "AT", "PROVIDE", "CHIP", "LIST", "SECT",
  "ABSOLUTE", "LOAD", "NEWLINE", "ENDWORD", "ORDER", "NAMEWORD", "FORMAT",
  "PUBLIC", "BASE", "ALIAS", "TRUNCATE", "REL", "INPUT_SCRIPT",
  "INPUT_MRI_SCRIPT", "INPUT_DEFSYM", "','", "';'", "')'", "'['", "']'",
  "'!'", "'~'", "$accept", "file", "filename", "defsym_expr", "$@1",
  "mri_script_file", "$@2", "mri_script_lines", "mri_script_command",
  "ordernamelist", "mri_load_name_list", "mri_abs_name_list",
  "script_file", "$@3", "ifile_list", "ifile_p1", "$@4", "$@5",
  "input_list", "sections", "sec_or_group_p1", "statement_anywhere",
  "file_NAME_list", "input_section_spec", "$@6", "$@7", "$@8", "statement",
  "statement_list", "statement_list_opt", "length", "fill_opt",
  "assign_op", "end", "assignment", "opt_comma", "memory",
  "memory_spec_list", "memory_spec", "$@9", "origin_spec", "length_spec",
  "attributes_opt", "startup", "high_level_library",
  "high_level_library_NAME_list", "low_level_library",
  "low_level_library_NAME_list", "floating_point_support", "mustbe_exp",
  "$@10", "exp", "opt_at", "section", "$@11", "$@12", "$@13", "$@14",
  "$@15", "type", "opt_exp_with_type", "memspec_opt", YY_NULL
};
#endif

# ifdef YYPRINT
/* YYTOKNUM[YYLEX-NUM] -- Internal token number corresponding to
   token YYLEX-NUM.  */
static const yytype_uint16 yytoknum[] =
{
       0,   256,   257,   258,   259,    61,   260,   261,   262,   263,
     264,   265,   266,   267,    63,    58,   268,   269,   124,    94,
      38,   270,   271,    60,    62,   272,   273,   274,   275,    43,
      45,    42,    47,    37,   276,   277,    40,   278,   279,   280,
     281,   282,   283,   284,   123,   125,   285,   286,   287,   288,
     289,   290,   291,   292,   293,   294,   295,   296,   297,   298,
     299,   300,   301,   302,   303,   304,   305,   306,   307,   308,
     309,   310,   311,   312,   313,   314,   315,   316,   317,   318,
     319,   320,   321,   322,   323,   324,   325,   326,   327,   328,
     329,   330,   331,   332,   333,   334,   335,   336,   337,   338,
      44,    59,    41,    91,    93,    33,   126
};
# endif

/* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_uint8 yyr1[] =
{
       0,   107,   108,   108,   108,   109,   111,   110,   113,   112,
     114,   114,   115,   115,   115,   115,   115,   115,   115,   115,
     115,   115,   115,   115,   115,   115,   115,   115,   115,   115,
     115,   115,   115,   115,   115,   116,   116,   116,   117,   117,
     118,   118,   120,   119,   121,   121,   122,   122,   122,   122,
     122,   122,   122,   122,   122,   122,   122,   122,   122,   122,
     122,   122,   123,   122,   122,   124,   122,   125,   125,   125,
     126,   127,   127,   127,   128,   128,   129,   129,   130,   131,
     130,   132,   130,   133,   130,   134,   134,   134,   134,   134,
     134,   134,   135,   135,   136,   136,   137,   137,   137,   137,
     138,   138,   139,   139,   139,   139,   139,   139,   139,   139,
     140,   140,   141,   141,   141,   142,   142,   143,   144,   144,
     144,   146,   145,   147,   148,   149,   149,   150,   151,   151,
     152,   152,   153,   154,   154,   155,   155,   157,   156,   158,
     158,   158,   158,   158,   158,   158,   158,   158,   158,   158,
     158,   158,   158,   158,   158,   158,   158,   158,   158,   158,
     158,   158,   158,   158,   158,   158,   158,   158,   158,   158,
     158,   158,   158,   159,   159,   161,   162,   163,   164,   165,
     160,   166,   166,   166,   166,   166,   166,   167,   167,   167,
     167,   168,   168
};

/* YYR2[YYN] -- Number of symbols composing right hand side of rule YYN.  */
static const yytype_uint8 yyr2[] =
{
       0,     2,     2,     2,     2,     1,     0,     4,     0,     2,
       3,     0,     2,     4,     1,     1,     2,     1,     4,     4,
       3,     2,     4,     3,     4,     4,     4,     2,     2,     2,
       4,     4,     2,     2,     0,     3,     2,     0,     1,     3,
       1,     3,     0,     2,     2,     0,     1,     1,     1,     1,
       1,     1,     1,     1,     4,     4,     4,     4,     8,     4,
       1,     4,     0,     5,     4,     0,     5,     1,     3,     2,
       4,     2,     2,     0,     4,     2,     1,     3,     1,     0,
       4,     0,     5,     0,     5,     2,     1,     1,     1,     1,
       4,     4,     2,     1,     0,     1,     1,     1,     1,     1,
       2,     0,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     3,     3,     6,     1,     0,     5,     2,     3,
       0,     0,     7,     3,     3,     3,     0,     4,     4,     3,
       3,     1,     4,     3,     0,     1,     1,     0,     2,     2,
       3,     4,     2,     2,     2,     3,     3,     3,     3,     3,
       3,     3,     3,     3,     3,     3,     3,     3,     3,     3,
       3,     5,     3,     3,     4,     1,     1,     4,     4,     4,
       4,     4,     1,     4,     0,     0,     0,     0,     0,     0,
      14,     1,     1,     1,     1,     1,     0,     2,     5,     1,
       4,     2,     0
};

/* YYDEFACT[STATE-NAME] -- Default reduction number in state STATE-NUM.
   Performed when YYTABLE doesn't specify something else to do.  Zero
   means the default is an error.  */
static const yytype_uint8 yydefact[] =
{
       0,    42,     8,     6,     0,     2,    45,     3,    11,     4,
       0,     1,    43,     9,     0,     0,     0,     0,    60,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,   135,
     136,     0,    62,     0,     0,    53,    44,    47,    52,     0,
      46,    48,    49,    50,    51,    14,     0,     0,     0,    15,
       0,     0,     0,    17,    37,     0,     0,     0,     0,     0,
       0,     0,     0,   137,   109,   108,   107,   106,   105,   104,
     103,   102,   137,    73,     0,     0,     5,    65,     0,     0,
       0,     0,     0,     0,     0,   134,     0,     0,     0,     0,
     111,   110,    75,     0,     0,   165,   172,     0,     0,     0,
       0,     0,   166,     0,     0,     0,     0,     0,     0,     0,
      12,     0,    40,    27,    38,    28,    16,    29,    21,     0,
      32,     0,    33,    10,     7,   112,     0,   113,     0,     0,
       0,    45,   121,   120,     0,     0,     0,     0,     0,   129,
     131,   116,   116,    67,     0,     0,     0,     0,     0,     0,
     143,   139,     0,     0,     0,     0,     0,     0,     0,     0,
     142,   144,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    23,     0,     0,    36,     0,     0,
       0,    20,     0,   138,   175,    70,    72,    71,     0,    57,
      59,     0,   126,     0,    54,    55,    64,    74,   127,   115,
     128,     0,   132,     0,    69,     0,    61,     0,    56,   137,
      25,    26,   140,     0,     0,     0,     0,     0,     0,     0,
       0,   163,   162,   160,   159,   158,   153,   152,   156,   157,
     155,   154,   151,   150,   148,   149,   145,   146,   147,    13,
      24,    22,    41,    39,    35,    18,    19,    31,    30,     0,
       0,    66,     0,     0,   117,     0,   118,   130,   133,    68,
      63,     0,   170,   171,   164,   167,   141,   168,   169,     0,
     189,   186,     0,   174,     0,     0,     0,   119,   114,   161,
     181,   182,   183,   184,   185,     0,   187,   186,     0,   176,
       0,   125,     0,   116,     0,     0,     0,     0,    58,   137,
       0,   190,     0,     0,   177,   123,     0,   122,   188,   173,
      94,   137,    78,    83,    96,    97,    98,    99,     0,    86,
      88,    87,    79,    89,    93,    95,     0,     0,     0,   124,
       0,     0,     0,     0,    92,   178,     0,    85,     0,     0,
       0,    76,   116,   192,     0,   116,   116,    91,    80,     0,
       0,   101,    90,    82,    84,    77,   191,   137,   179,   100,
     116,   180
};

/* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,     4,    77,     9,    10,     7,     8,    13,    61,   116,
     115,   113,     5,     6,    12,    36,    87,   131,   144,    37,
     128,    38,   352,   333,   343,   340,   341,   334,   335,   336,
     337,   368,    72,    92,    39,   359,    40,   203,   133,   202,
     303,   317,   263,    41,    42,   141,    43,   142,    44,   125,
     126,   152,   299,   197,   259,   307,   320,   353,   370,   295,
     283,   361
};

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
#define YYPACT_NINF -285
static const yytype_int16 yypact[] =
{
     -23,  -285,  -285,  -285,    17,  -285,  -285,  -285,  -285,  -285,
      36,  -285,   362,   363,    54,   762,    58,    71,  -285,    76,
     136,   106,   120,   126,   128,   130,   131,   132,   135,  -285,
    -285,   137,  -285,   138,   139,  -285,  -285,  -285,  -285,    52,
    -285,  -285,  -285,  -285,  -285,  -285,   168,   181,   263,  -285,
     183,   184,   185,  -285,  -285,   195,   197,   198,   263,   204,
     206,   123,   263,  -285,  -285,  -285,  -285,  -285,  -285,  -285,
    -285,  -285,  -285,  -285,   212,   217,  -285,  -285,   218,   220,
     136,   136,   222,   136,    11,  -285,   223,   192,   136,   225,
    -285,  -285,  -285,   228,   231,  -285,  -285,   263,   263,   263,
     194,   207,  -285,   208,   211,   213,   215,   219,   263,   263,
      63,   154,  -285,   148,  -285,   157,     3,  -285,  -285,   177,
     622,   163,  -285,  -285,   622,  -285,   263,  -285,   175,   -63,
     166,  -285,  -285,  -285,   171,   172,   173,   174,   178,  -285,
    -285,   -38,   -30,  -285,     8,   223,   179,   260,   263,   263,
    -285,  -285,     2,   263,   263,   280,   284,   263,   285,   263,
    -285,  -285,   263,   263,   263,   263,   263,   263,   263,   263,
     263,   263,   263,   263,   263,   263,   263,   263,   263,   263,
     263,   263,   263,   263,   622,   286,   287,  -285,   293,   263,
     263,   622,   151,   622,   762,  -285,  -285,  -285,   299,  -285,
    -285,   314,   269,    -2,  -285,  -285,  -285,  -285,  -285,  -285,
    -285,   136,  -285,   136,  -285,   302,  -285,    37,  -285,  -285,
     622,   622,  -285,   448,   468,   214,   221,   488,   227,   508,
     664,   641,   680,   695,   709,   722,   733,   733,    25,    25,
      25,    25,    15,    15,   114,   114,  -285,  -285,  -285,   622,
     622,   622,  -285,  -285,  -285,   622,   622,  -285,  -285,   249,
     210,  -285,   311,   304,  -285,   218,  -285,  -285,  -285,  -285,
    -285,   233,  -285,  -285,  -285,  -285,  -285,  -285,  -285,   263,
    -285,   140,   597,   237,   316,   234,   251,  -285,  -285,   622,
    -285,  -285,  -285,  -285,  -285,   235,  -285,    12,   288,  -285,
     236,  -285,   320,   232,   315,   248,   263,   307,  -285,  -285,
     279,  -285,   338,   535,  -285,  -285,   351,  -285,  -285,  -285,
     230,  -285,   334,  -285,  -285,  -285,  -285,  -285,   322,  -285,
    -285,  -285,  -285,  -285,  -285,   230,   326,   323,    52,  -285,
     324,   336,   263,   373,  -285,  -285,   263,  -285,   373,   373,
     555,  -285,   -62,   354,   575,    -3,    49,  -285,  -285,   375,
     381,   382,  -285,  -285,  -285,  -285,  -285,  -285,  -285,  -285,
     232,  -285
};

/* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -285,  -285,   -75,  -285,  -285,  -285,  -285,  -285,  -285,  -285,
    -285,  -285,  -285,  -285,   255,  -285,  -285,  -285,   243,  -285,
    -285,   264,  -188,  -285,  -285,  -285,  -285,    59,  -285,  -285,
    -285,  -285,  -285,    55,  -284,  -138,  -285,  -285,  -192,  -285,
    -285,  -285,  -285,  -285,  -285,  -285,  -285,  -285,  -285,   -71,
    -285,   -48,  -285,  -285,  -285,  -285,  -285,  -285,  -285,    99,
    -285,  -285
};

/* YYTABLE[YYPACT[STATE-NUM]].  What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule which
   number is the opposite.  If YYTABLE_NINF, syntax error.  */
#define YYTABLE_NINF -82
static const yytype_int16 yytable[] =
{
     110,   127,   132,   211,   213,   135,   136,   187,   138,   140,
     120,   266,   214,   146,   124,    76,   162,    11,   163,   164,
     165,   166,   167,   168,   169,   170,   171,   172,   173,   174,
     175,   176,   177,   178,   179,   180,   338,   198,   209,   199,
      14,   214,   358,   264,   176,   177,   178,   179,   180,   150,
     151,   338,   174,   175,   176,   177,   178,   179,   180,    62,
     160,   161,   209,   184,   210,   290,   291,   292,   293,   294,
     209,   191,   212,   287,     1,     2,     3,   162,   193,   163,
     164,   165,   166,   167,   168,   169,   170,   171,   172,   173,
     174,   175,   176,   177,   178,   179,   180,   209,   265,   363,
     220,   221,    73,   188,   222,   223,   224,    74,   215,   227,
     216,   229,    75,   139,   230,   231,   232,   233,   234,   235,
     236,   237,   238,   239,   240,   241,   242,   243,   244,   245,
     246,   247,   248,   249,   250,   251,   267,   215,   268,   270,
      76,   255,   256,    95,    96,   178,   179,   180,   271,   209,
      78,   364,    90,    91,   257,   258,    79,    95,    96,   182,
     355,   356,    80,   181,    81,   310,    82,    83,    84,    97,
      98,    85,    93,    86,    88,    89,    99,   100,   101,   194,
      95,    96,   189,    97,    98,    94,   102,   111,   112,   114,
      99,   100,   101,   290,   291,   292,   293,   294,   103,   117,
     102,   118,   119,   104,   105,   106,    97,    98,   121,   122,
     123,   282,   103,    99,   100,   101,   129,   104,   105,   106,
     195,   130,   132,   102,   134,   107,   137,   143,   145,   147,
     153,   289,   371,   148,   322,   103,   149,    25,   315,   107,
     104,   105,   106,   154,   155,   108,   109,   156,   185,   157,
     339,   158,    95,    96,   183,   159,    34,   186,   313,   108,
     109,   323,   107,   192,   280,   219,    95,    96,   200,   324,
     325,   326,   327,   204,   205,   206,   207,   190,    97,    98,
     208,   218,   108,   109,   225,   281,   100,   101,   226,   228,
     252,   253,    97,    98,   350,   102,   369,   254,   354,    99,
     100,   101,   328,   260,   329,   262,   269,   103,   330,   102,
     284,    34,   104,   105,   106,   285,   274,   298,    15,   286,
     300,   103,   302,   275,   306,   309,   104,   105,   106,   277,
     311,   331,   209,   332,   107,   288,   301,   304,   308,    63,
      64,    65,    66,    67,    68,    69,    70,    71,   107,   261,
     312,   314,   316,   318,   108,   109,   321,    16,   342,   346,
     348,    17,    18,    19,    20,    21,    15,    45,   108,   109,
     -81,   345,   349,    22,    23,    24,    25,   351,   360,   365,
      26,    27,    28,    29,    30,   366,   201,   367,   217,    31,
      32,    33,   196,   347,   344,    34,   305,     0,     0,     0,
      46,     0,     0,     0,     0,    16,     0,     0,     0,    17,
      18,    19,    20,    21,     0,    35,     0,     0,     0,     0,
       0,    22,    23,    24,    25,     0,     0,     0,    26,    27,
      28,    29,    30,     0,     0,     0,     0,    31,    32,    33,
       0,     0,    47,    34,     0,    48,    49,    50,    51,    52,
     -34,    53,    54,    55,    56,    57,    58,    59,    60,     0,
       0,     0,   162,    35,   163,   164,   165,   166,   167,   168,
     169,   170,   171,   172,   173,   174,   175,   176,   177,   178,
     179,   180,   162,     0,   163,   164,   165,   166,   167,   168,
     169,   170,   171,   172,   173,   174,   175,   176,   177,   178,
     179,   180,   162,     0,   163,   164,   165,   166,   167,   168,
     169,   170,   171,   172,   173,   174,   175,   176,   177,   178,
     179,   180,   162,     0,   163,   164,   165,   166,   167,   168,
     169,   170,   171,   172,   173,   174,   175,   176,   177,   178,
     179,   180,     0,     0,     0,     0,     0,     0,     0,   162,
     272,   163,   164,   165,   166,   167,   168,   169,   170,   171,
     172,   173,   174,   175,   176,   177,   178,   179,   180,   162,
     273,   163,   164,   165,   166,   167,   168,   169,   170,   171,
     172,   173,   174,   175,   176,   177,   178,   179,   180,   162,
     276,   163,   164,   165,   166,   167,   168,   169,   170,   171,
     172,   173,   174,   175,   176,   177,   178,   179,   180,     0,
     278,   162,   296,   163,   164,   165,   166,   167,   168,   169,
     170,   171,   172,   173,   174,   175,   176,   177,   178,   179,
     180,     0,     0,   297,     0,     0,   162,   319,   163,   164,
     165,   166,   167,   168,   169,   170,   171,   172,   173,   174,
     175,   176,   177,   178,   179,   180,     0,   357,   164,   165,
     166,   167,   168,   169,   170,   171,   172,   173,   174,   175,
     176,   177,   178,   179,   180,     0,     0,   362,   162,   279,
     163,   164,   165,   166,   167,   168,   169,   170,   171,   172,
     173,   174,   175,   176,   177,   178,   179,   180,   165,   166,
     167,   168,   169,   170,   171,   172,   173,   174,   175,   176,
     177,   178,   179,   180,   166,   167,   168,   169,   170,   171,
     172,   173,   174,   175,   176,   177,   178,   179,   180,   167,
     168,   169,   170,   171,   172,   173,   174,   175,   176,   177,
     178,   179,   180,   168,   169,   170,   171,   172,   173,   174,
     175,   176,   177,   178,   179,   180,   170,   171,   172,   173,
     174,   175,   176,   177,   178,   179,   180,    63,    64,    65,
      66,    67,    68,    69,    70,    71
};

#define yypact_value_is_default(Yystate) \
  (!!((Yystate) == (-285)))

#define yytable_value_is_error(Yytable_value) \
  YYID (0)

static const yytype_int16 yycheck[] =
{
      48,    72,     4,   141,   142,    80,    81,     4,    83,    84,
      58,   203,     4,    88,    62,     4,    14,     0,    16,    17,
      18,    19,    20,    21,    22,    23,    24,    25,    26,    27,
      28,    29,    30,    31,    32,    33,   320,   100,   100,   102,
       4,     4,   104,    45,    29,    30,    31,    32,    33,    97,
      98,   335,    27,    28,    29,    30,    31,    32,    33,     5,
     108,   109,   100,   111,   102,    53,    54,    55,    56,    57,
     100,   119,   102,   265,    97,    98,    99,    14,   126,    16,
      17,    18,    19,    20,    21,    22,    23,    24,    25,    26,
      27,    28,    29,    30,    31,    32,    33,   100,   100,   102,
     148,   149,    44,   100,   102,   153,   154,    36,   100,   157,
     102,   159,    36,   102,   162,   163,   164,   165,   166,   167,
     168,   169,   170,   171,   172,   173,   174,   175,   176,   177,
     178,   179,   180,   181,   182,   183,   211,   100,   213,   102,
       4,   189,   190,     3,     4,    31,    32,    33,   219,   100,
      44,   102,   100,   101,     3,     4,    36,     3,     4,     5,
     348,   349,    36,   100,    36,   303,    36,    36,    36,    29,
      30,    36,     4,    36,    36,    36,    36,    37,    38,     4,
       3,     4,     5,    29,    30,     4,    46,     4,     4,     4,
      36,    37,    38,    53,    54,    55,    56,    57,    58,     4,
      46,     4,     4,    63,    64,    65,    29,    30,     4,     3,
      87,   259,    58,    36,    37,    38,     4,    63,    64,    65,
      45,     4,     4,    46,     4,    85,     4,     4,    36,     4,
      36,   279,   370,     5,     4,    58,     5,    62,   309,    85,
      63,    64,    65,    36,    36,   105,   106,    36,   100,    36,
     321,    36,     3,     4,   100,    36,    81,   100,   306,   105,
     106,    31,    85,   100,    15,     5,     3,     4,   102,    39,
      40,    41,    42,   102,   102,   102,   102,   100,    29,    30,
     102,   102,   105,   106,     4,    36,    37,    38,     4,     4,
       4,     4,    29,    30,   342,    46,   367,     4,   346,    36,
      37,    38,    72,     4,    74,    36,     4,    58,    78,    46,
     100,    81,    63,    64,    65,     4,   102,    80,     4,    15,
       4,    58,    71,   102,    36,     5,    63,    64,    65,   102,
      15,   101,   100,   103,    85,   102,   102,   102,   102,     5,
       6,     7,     8,     9,    10,    11,    12,    13,    85,    35,
     102,    44,    73,    15,   105,   106,     5,    43,    36,    36,
      36,    47,    48,    49,    50,    51,     4,     4,   105,   106,
      36,    45,    36,    59,    60,    61,    62,     4,    24,     4,
      66,    67,    68,    69,    70,     4,   131,     5,   145,    75,
      76,    77,   128,   338,   335,    81,   297,    -1,    -1,    -1,
      37,    -1,    -1,    -1,    -1,    43,    -1,    -1,    -1,    47,
      48,    49,    50,    51,    -1,   101,    -1,    -1,    -1,    -1,
      -1,    59,    60,    61,    62,    -1,    -1,    -1,    66,    67,
      68,    69,    70,    -1,    -1,    -1,    -1,    75,    76,    77,
      -1,    -1,    79,    81,    -1,    82,    83,    84,    85,    86,
      87,    88,    89,    90,    91,    92,    93,    94,    95,    -1,
      -1,    -1,    14,   101,    16,    17,    18,    19,    20,    21,
      22,    23,    24,    25,    26,    27,    28,    29,    30,    31,
      32,    33,    14,    -1,    16,    17,    18,    19,    20,    21,
      22,    23,    24,    25,    26,    27,    28,    29,    30,    31,
      32,    33,    14,    -1,    16,    17,    18,    19,    20,    21,
      22,    23,    24,    25,    26,    27,    28,    29,    30,    31,
      32,    33,    14,    -1,    16,    17,    18,    19,    20,    21,
      22,    23,    24,    25,    26,    27,    28,    29,    30,    31,
      32,    33,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    14,
     102,    16,    17,    18,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    30,    31,    32,    33,    14,
     102,    16,    17,    18,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    30,    31,    32,    33,    14,
     102,    16,    17,    18,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    30,    31,    32,    33,    -1,
     102,    14,    15,    16,    17,    18,    19,    20,    21,    22,
      23,    24,    25,    26,    27,    28,    29,    30,    31,    32,
      33,    -1,    -1,    36,    -1,    -1,    14,   102,    16,    17,
      18,    19,    20,    21,    22,    23,    24,    25,    26,    27,
      28,    29,    30,    31,    32,    33,    -1,   102,    17,    18,
      19,    20,    21,    22,    23,    24,    25,    26,    27,    28,
      29,    30,    31,    32,    33,    -1,    -1,   102,    14,    15,
      16,    17,    18,    19,    20,    21,    22,    23,    24,    25,
      26,    27,    28,    29,    30,    31,    32,    33,    18,    19,
      20,    21,    22,    23,    24,    25,    26,    27,    28,    29,
      30,    31,    32,    33,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    30,    31,    32,    33,    20,
      21,    22,    23,    24,    25,    26,    27,    28,    29,    30,
      31,    32,    33,    21,    22,    23,    24,    25,    26,    27,
      28,    29,    30,    31,    32,    33,    23,    24,    25,    26,
      27,    28,    29,    30,    31,    32,    33,     5,     6,     7,
       8,     9,    10,    11,    12,    13
};

/* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
   symbol of state STATE-NUM.  */
static const yytype_uint8 yystos[] =
{
       0,    97,    98,    99,   108,   119,   120,   112,   113,   110,
     111,     0,   121,   114,     4,     4,    43,    47,    48,    49,
      50,    51,    59,    60,    61,    62,    66,    67,    68,    69,
      70,    75,    76,    77,    81,   101,   122,   126,   128,   141,
     143,   150,   151,   153,   155,     4,    37,    79,    82,    83,
      84,    85,    86,    88,    89,    90,    91,    92,    93,    94,
      95,   115,     5,     5,     6,     7,     8,     9,    10,    11,
      12,    13,   139,    44,    36,    36,     4,   109,    44,    36,
      36,    36,    36,    36,    36,    36,    36,   123,    36,    36,
     100,   101,   140,     4,     4,     3,     4,    29,    30,    36,
      37,    38,    46,    58,    63,    64,    65,    85,   105,   106,
     158,     4,     4,   118,     4,   117,   116,     4,     4,     4,
     158,     4,     3,    87,   158,   156,   157,   156,   127,     4,
       4,   124,     4,   145,     4,   109,   109,     4,   109,   102,
     109,   152,   154,     4,   125,    36,   109,     4,     5,     5,
     158,   158,   158,    36,    36,    36,    36,    36,    36,    36,
     158,   158,    14,    16,    17,    18,    19,    20,    21,    22,
      23,    24,    25,    26,    27,    28,    29,    30,    31,    32,
      33,   100,     5,   100,   158,   100,   100,     4,   100,     5,
     100,   158,   100,   158,     4,    45,   128,   160,   100,   102,
     102,   121,   146,   144,   102,   102,   102,   102,   102,   100,
     102,   142,   102,   142,     4,   100,   102,   125,   102,     5,
     158,   158,   102,   158,   158,     4,     4,   158,     4,   158,
     158,   158,   158,   158,   158,   158,   158,   158,   158,   158,
     158,   158,   158,   158,   158,   158,   158,   158,   158,   158,
     158,   158,     4,     4,     4,   158,   158,     3,     4,   161,
       4,    35,    36,   149,    45,   100,   145,   109,   109,     4,
     102,   156,   102,   102,   102,   102,   102,   102,   102,    15,
      15,    36,   158,   167,   100,     4,    15,   145,   102,   158,
      53,    54,    55,    56,    57,   166,    15,    36,    80,   159,
       4,   102,    71,   147,   102,   166,    36,   162,   102,     5,
     142,    15,   102,   158,    44,   156,    73,   148,    15,   102,
     163,     5,     4,    31,    39,    40,    41,    42,    72,    74,
      78,   101,   103,   130,   134,   135,   136,   137,   141,   156,
     132,   133,    36,   131,   134,    45,    36,   140,    36,    36,
     158,     4,   129,   164,   158,   129,   129,   102,   104,   142,
      24,   168,   102,   102,   102,     4,     4,     5,   138,   156,
     165,   142
};

#define yyerrok		(yyerrstatus = 0)
#define yyclearin	(yychar = YYEMPTY)
#define YYEMPTY		(-2)
#define YYEOF		0

#define YYACCEPT	goto yyacceptlab
#define YYABORT		goto yyabortlab
#define YYERROR		goto yyerrorlab


/* Like YYERROR except do call yyerror.  This remains here temporarily
   to ease the transition to the new meaning of YYERROR, for GCC.
   Once GCC version 2 has supplanted version 1, this can go.  However,
   YYFAIL appears to be in use.  Nevertheless, it is formally deprecated
   in Bison 2.4.2's NEWS entry, where a plan to phase it out is
   discussed.  */

#define YYFAIL		goto yyerrlab
#if defined YYFAIL
  /* This is here to suppress warnings from the GCC cpp's
     -Wunused-macros.  Normally we don't worry about that warning, but
     some users do, and we want to make it easy for users to remove
     YYFAIL uses, which will produce warnings from Bison 2.5.  */
#endif

#define YYRECOVERING()  (!!yyerrstatus)

#define YYBACKUP(Token, Value)                                  \
do                                                              \
  if (yychar == YYEMPTY)                                        \
    {                                                           \
      yychar = (Token);                                         \
      yylval = (Value);                                         \
      YYPOPSTACK (yylen);                                       \
      yystate = *yyssp;                                         \
      goto yybackup;                                            \
    }                                                           \
  else                                                          \
    {                                                           \
      yyerror (YY_("syntax error: cannot back up")); \
      YYERROR;							\
    }								\
while (YYID (0))

/* Error token number */
#define YYTERROR	1
#define YYERRCODE	256


/* This macro is provided for backward compatibility. */
#ifndef YY_LOCATION_PRINT
# define YY_LOCATION_PRINT(File, Loc) ((void) 0)
#endif


/* YYLEX -- calling `yylex' with the right arguments.  */
#ifdef YYLEX_PARAM
# define YYLEX yylex (YYLEX_PARAM)
#else
# define YYLEX yylex ()
#endif

/* Enable debugging if requested.  */
#if YYDEBUG

# ifndef YYFPRINTF
#  include <stdio.h> /* INFRINGES ON USER NAME SPACE */
#  define YYFPRINTF fprintf
# endif

# define YYDPRINTF(Args)			\
do {						\
  if (yydebug)					\
    YYFPRINTF Args;				\
} while (YYID (0))

# define YY_SYMBOL_PRINT(Title, Type, Value, Location)			  \
do {									  \
  if (yydebug)								  \
    {									  \
      YYFPRINTF (stderr, "%s ", Title);					  \
      yy_symbol_print (stderr,						  \
		  Type, Value); \
      YYFPRINTF (stderr, "\n");						  \
    }									  \
} while (YYID (0))


/*--------------------------------.
| Print this symbol on YYOUTPUT.  |
`--------------------------------*/

/*ARGSUSED*/
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yy_symbol_value_print (FILE *yyoutput, int yytype, YYSTYPE const * const yyvaluep)
#else
static void
yy_symbol_value_print (yyoutput, yytype, yyvaluep)
    FILE *yyoutput;
    int yytype;
    YYSTYPE const * const yyvaluep;
#endif
{
  FILE *yyo = yyoutput;
  YYUSE (yyo);
  if (!yyvaluep)
    return;
# ifdef YYPRINT
  if (yytype < YYNTOKENS)
    YYPRINT (yyoutput, yytoknum[yytype], *yyvaluep);
# else
  YYUSE (yyoutput);
# endif
  switch (yytype)
    {
      default:
        break;
    }
}


/*--------------------------------.
| Print this symbol on YYOUTPUT.  |
`--------------------------------*/

#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yy_symbol_print (FILE *yyoutput, int yytype, YYSTYPE const * const yyvaluep)
#else
static void
yy_symbol_print (yyoutput, yytype, yyvaluep)
    FILE *yyoutput;
    int yytype;
    YYSTYPE const * const yyvaluep;
#endif
{
  if (yytype < YYNTOKENS)
    YYFPRINTF (yyoutput, "token %s (", yytname[yytype]);
  else
    YYFPRINTF (yyoutput, "nterm %s (", yytname[yytype]);

  yy_symbol_value_print (yyoutput, yytype, yyvaluep);
  YYFPRINTF (yyoutput, ")");
}

/*------------------------------------------------------------------.
| yy_stack_print -- Print the state stack from its BOTTOM up to its |
| TOP (included).                                                   |
`------------------------------------------------------------------*/

#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yy_stack_print (yytype_int16 *yybottom, yytype_int16 *yytop)
#else
static void
yy_stack_print (yybottom, yytop)
    yytype_int16 *yybottom;
    yytype_int16 *yytop;
#endif
{
  YYFPRINTF (stderr, "Stack now");
  for (; yybottom <= yytop; yybottom++)
    {
      int yybot = *yybottom;
      YYFPRINTF (stderr, " %d", yybot);
    }
  YYFPRINTF (stderr, "\n");
}

# define YY_STACK_PRINT(Bottom, Top)				\
do {								\
  if (yydebug)							\
    yy_stack_print ((Bottom), (Top));				\
} while (YYID (0))


/*------------------------------------------------.
| Report that the YYRULE is going to be reduced.  |
`------------------------------------------------*/

#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yy_reduce_print (YYSTYPE *yyvsp, int yyrule)
#else
static void
yy_reduce_print (yyvsp, yyrule)
    YYSTYPE *yyvsp;
    int yyrule;
#endif
{
  int yynrhs = yyr2[yyrule];
  int yyi;
  unsigned long int yylno = yyrline[yyrule];
  YYFPRINTF (stderr, "Reducing stack by rule %d (line %lu):\n",
	     yyrule - 1, yylno);
  /* The symbols being reduced.  */
  for (yyi = 0; yyi < yynrhs; yyi++)
    {
      YYFPRINTF (stderr, "   $%d = ", yyi + 1);
      yy_symbol_print (stderr, yyrhs[yyprhs[yyrule] + yyi],
		       &(yyvsp[(yyi + 1) - (yynrhs)])
		       		       );
      YYFPRINTF (stderr, "\n");
    }
}

# define YY_REDUCE_PRINT(Rule)		\
do {					\
  if (yydebug)				\
    yy_reduce_print (yyvsp, Rule); \
} while (YYID (0))

/* Nonzero means print parse trace.  It is left uninitialized so that
   multiple parsers can coexist.  */
int yydebug;
#else /* !YYDEBUG */
# define YYDPRINTF(Args)
# define YY_SYMBOL_PRINT(Title, Type, Value, Location)
# define YY_STACK_PRINT(Bottom, Top)
# define YY_REDUCE_PRINT(Rule)
#endif /* !YYDEBUG */


/* YYINITDEPTH -- initial size of the parser's stacks.  */
#ifndef	YYINITDEPTH
# define YYINITDEPTH 200
#endif

/* YYMAXDEPTH -- maximum size the stacks can grow to (effective only
   if the built-in stack extension method is used).

   Do not make this value too large; the results are undefined if
   YYSTACK_ALLOC_MAXIMUM < YYSTACK_BYTES (YYMAXDEPTH)
   evaluated with infinite-precision integer arithmetic.  */

#ifndef YYMAXDEPTH
# define YYMAXDEPTH 10000
#endif


#if YYERROR_VERBOSE

# ifndef yystrlen
#  if defined __GLIBC__ && defined _STRING_H
#   define yystrlen strlen
#  else
/* Return the length of YYSTR.  */
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static YYSIZE_T
yystrlen (const char *yystr)
#else
static YYSIZE_T
yystrlen (yystr)
    const char *yystr;
#endif
{
  YYSIZE_T yylen;
  for (yylen = 0; yystr[yylen]; yylen++)
    continue;
  return yylen;
}
#  endif
# endif

# ifndef yystpcpy
#  if defined __GLIBC__ && defined _STRING_H && defined _GNU_SOURCE
#   define yystpcpy stpcpy
#  else
/* Copy YYSRC to YYDEST, returning the address of the terminating '\0' in
   YYDEST.  */
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static char *
yystpcpy (char *yydest, const char *yysrc)
#else
static char *
yystpcpy (yydest, yysrc)
    char *yydest;
    const char *yysrc;
#endif
{
  char *yyd = yydest;
  const char *yys = yysrc;

  while ((*yyd++ = *yys++) != '\0')
    continue;

  return yyd - 1;
}
#  endif
# endif

# ifndef yytnamerr
/* Copy to YYRES the contents of YYSTR after stripping away unnecessary
   quotes and backslashes, so that it's suitable for yyerror.  The
   heuristic is that double-quoting is unnecessary unless the string
   contains an apostrophe, a comma, or backslash (other than
   backslash-backslash).  YYSTR is taken from yytname.  If YYRES is
   null, do not copy; instead, return the length of what the result
   would have been.  */
static YYSIZE_T
yytnamerr (char *yyres, const char *yystr)
{
  if (*yystr == '"')
    {
      YYSIZE_T yyn = 0;
      char const *yyp = yystr;

      for (;;)
	switch (*++yyp)
	  {
	  case '\'':
	  case ',':
	    goto do_not_strip_quotes;

	  case '\\':
	    if (*++yyp != '\\')
	      goto do_not_strip_quotes;
	    /* Fall through.  */
	  default:
	    if (yyres)
	      yyres[yyn] = *yyp;
	    yyn++;
	    break;

	  case '"':
	    if (yyres)
	      yyres[yyn] = '\0';
	    return yyn;
	  }
    do_not_strip_quotes: ;
    }

  if (! yyres)
    return yystrlen (yystr);

  return yystpcpy (yyres, yystr) - yyres;
}
# endif

/* Copy into *YYMSG, which is of size *YYMSG_ALLOC, an error message
   about the unexpected token YYTOKEN for the state stack whose top is
   YYSSP.

   Return 0 if *YYMSG was successfully written.  Return 1 if *YYMSG is
   not large enough to hold the message.  In that case, also set
   *YYMSG_ALLOC to the required number of bytes.  Return 2 if the
   required number of bytes is too large to store.  */
static int
yysyntax_error (YYSIZE_T *yymsg_alloc, char **yymsg,
                yytype_int16 *yyssp, int yytoken)
{
  YYSIZE_T yysize0 = yytnamerr (YY_NULL, yytname[yytoken]);
  YYSIZE_T yysize = yysize0;
  enum { YYERROR_VERBOSE_ARGS_MAXIMUM = 5 };
  /* Internationalized format string. */
  const char *yyformat = YY_NULL;
  /* Arguments of yyformat. */
  char const *yyarg[YYERROR_VERBOSE_ARGS_MAXIMUM];
  /* Number of reported tokens (one for the "unexpected", one per
     "expected"). */
  int yycount = 0;

  /* There are many possibilities here to consider:
     - Assume YYFAIL is not used.  It's too flawed to consider.  See
       <http://lists.gnu.org/archive/html/bison-patches/2009-12/msg00024.html>
       for details.  YYERROR is fine as it does not invoke this
       function.
     - If this state is a consistent state with a default action, then
       the only way this function was invoked is if the default action
       is an error action.  In that case, don't check for expected
       tokens because there are none.
     - The only way there can be no lookahead present (in yychar) is if
       this state is a consistent state with a default action.  Thus,
       detecting the absence of a lookahead is sufficient to determine
       that there is no unexpected or expected token to report.  In that
       case, just report a simple "syntax error".
     - Don't assume there isn't a lookahead just because this state is a
       consistent state with a default action.  There might have been a
       previous inconsistent state, consistent state with a non-default
       action, or user semantic action that manipulated yychar.
     - Of course, the expected token list depends on states to have
       correct lookahead information, and it depends on the parser not
       to perform extra reductions after fetching a lookahead from the
       scanner and before detecting a syntax error.  Thus, state merging
       (from LALR or IELR) and default reductions corrupt the expected
       token list.  However, the list is correct for canonical LR with
       one exception: it will still contain any token that will not be
       accepted due to an error action in a later state.
  */
  if (yytoken != YYEMPTY)
    {
      int yyn = yypact[*yyssp];
      yyarg[yycount++] = yytname[yytoken];
      if (!yypact_value_is_default (yyn))
        {
          /* Start YYX at -YYN if negative to avoid negative indexes in
             YYCHECK.  In other words, skip the first -YYN actions for
             this state because they are default actions.  */
          int yyxbegin = yyn < 0 ? -yyn : 0;
          /* Stay within bounds of both yycheck and yytname.  */
          int yychecklim = YYLAST - yyn + 1;
          int yyxend = yychecklim < YYNTOKENS ? yychecklim : YYNTOKENS;
          int yyx;

          for (yyx = yyxbegin; yyx < yyxend; ++yyx)
            if (yycheck[yyx + yyn] == yyx && yyx != YYTERROR
                && !yytable_value_is_error (yytable[yyx + yyn]))
              {
                if (yycount == YYERROR_VERBOSE_ARGS_MAXIMUM)
                  {
                    yycount = 1;
                    yysize = yysize0;
                    break;
                  }
                yyarg[yycount++] = yytname[yyx];
                {
                  YYSIZE_T yysize1 = yysize + yytnamerr (YY_NULL, yytname[yyx]);
                  if (! (yysize <= yysize1
                         && yysize1 <= YYSTACK_ALLOC_MAXIMUM))
                    return 2;
                  yysize = yysize1;
                }
              }
        }
    }

  switch (yycount)
    {
# define YYCASE_(N, S)                      \
      case N:                               \
        yyformat = S;                       \
      break
      YYCASE_(0, YY_("syntax error"));
      YYCASE_(1, YY_("syntax error, unexpected %s"));
      YYCASE_(2, YY_("syntax error, unexpected %s, expecting %s"));
      YYCASE_(3, YY_("syntax error, unexpected %s, expecting %s or %s"));
      YYCASE_(4, YY_("syntax error, unexpected %s, expecting %s or %s or %s"));
      YYCASE_(5, YY_("syntax error, unexpected %s, expecting %s or %s or %s or %s"));
# undef YYCASE_
    }

  {
    YYSIZE_T yysize1 = yysize + yystrlen (yyformat);
    if (! (yysize <= yysize1 && yysize1 <= YYSTACK_ALLOC_MAXIMUM))
      return 2;
    yysize = yysize1;
  }

  if (*yymsg_alloc < yysize)
    {
      *yymsg_alloc = 2 * yysize;
      if (! (yysize <= *yymsg_alloc
             && *yymsg_alloc <= YYSTACK_ALLOC_MAXIMUM))
        *yymsg_alloc = YYSTACK_ALLOC_MAXIMUM;
      return 1;
    }

  /* Avoid sprintf, as that infringes on the user's name space.
     Don't have undefined behavior even if the translation
     produced a string with the wrong number of "%s"s.  */
  {
    char *yyp = *yymsg;
    int yyi = 0;
    while ((*yyp = *yyformat) != '\0')
      if (*yyp == '%' && yyformat[1] == 's' && yyi < yycount)
        {
          yyp += yytnamerr (yyp, yyarg[yyi++]);
          yyformat += 2;
        }
      else
        {
          yyp++;
          yyformat++;
        }
  }
  return 0;
}
#endif /* YYERROR_VERBOSE */

/*-----------------------------------------------.
| Release the memory associated to this symbol.  |
`-----------------------------------------------*/

/*ARGSUSED*/
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yydestruct (const char *yymsg, int yytype, YYSTYPE *yyvaluep)
#else
static void
yydestruct (yymsg, yytype, yyvaluep)
    const char *yymsg;
    int yytype;
    YYSTYPE *yyvaluep;
#endif
{
  YYUSE (yyvaluep);

  if (!yymsg)
    yymsg = "Deleting";
  YY_SYMBOL_PRINT (yymsg, yytype, yyvaluep, yylocationp);

  switch (yytype)
    {

      default:
        break;
    }
}




/* The lookahead symbol.  */
int yychar;


#ifndef YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
# define YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
# define YY_IGNORE_MAYBE_UNINITIALIZED_END
#endif
#ifndef YY_INITIAL_VALUE
# define YY_INITIAL_VALUE(Value) /* Nothing. */
#endif

/* The semantic value of the lookahead symbol.  */
YYSTYPE yylval YY_INITIAL_VALUE(yyval_default);

/* Number of syntax errors so far.  */
int yynerrs;


/*----------.
| yyparse.  |
`----------*/

#ifdef YYPARSE_PARAM
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
int
yyparse (void *YYPARSE_PARAM)
#else
int
yyparse (YYPARSE_PARAM)
    void *YYPARSE_PARAM;
#endif
#else /* ! YYPARSE_PARAM */
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
int
yyparse (void)
#else
int
yyparse ()

#endif
#endif
{
    int yystate;
    /* Number of tokens to shift before error messages enabled.  */
    int yyerrstatus;

    /* The stacks and their tools:
       `yyss': related to states.
       `yyvs': related to semantic values.

       Refer to the stacks through separate pointers, to allow yyoverflow
       to reallocate them elsewhere.  */

    /* The state stack.  */
    yytype_int16 yyssa[YYINITDEPTH];
    yytype_int16 *yyss;
    yytype_int16 *yyssp;

    /* The semantic value stack.  */
    YYSTYPE yyvsa[YYINITDEPTH];
    YYSTYPE *yyvs;
    YYSTYPE *yyvsp;

    YYSIZE_T yystacksize;

  int yyn;
  int yyresult;
  /* Lookahead token as an internal (translated) token number.  */
  int yytoken = 0;
  /* The variables used to return semantic value and location from the
     action routines.  */
  YYSTYPE yyval;

#if YYERROR_VERBOSE
  /* Buffer for error messages, and its allocated size.  */
  char yymsgbuf[128];
  char *yymsg = yymsgbuf;
  YYSIZE_T yymsg_alloc = sizeof yymsgbuf;
#endif

#define YYPOPSTACK(N)   (yyvsp -= (N), yyssp -= (N))

  /* The number of symbols on the RHS of the reduced rule.
     Keep to zero when no symbol should be popped.  */
  int yylen = 0;

  yyssp = yyss = yyssa;
  yyvsp = yyvs = yyvsa;
  yystacksize = YYINITDEPTH;

  YYDPRINTF ((stderr, "Starting parse\n"));

  yystate = 0;
  yyerrstatus = 0;
  yynerrs = 0;
  yychar = YYEMPTY; /* Cause a token to be read.  */
  goto yysetstate;

/*------------------------------------------------------------.
| yynewstate -- Push a new state, which is found in yystate.  |
`------------------------------------------------------------*/
 yynewstate:
  /* In all cases, when you get here, the value and location stacks
     have just been pushed.  So pushing a state here evens the stacks.  */
  yyssp++;

 yysetstate:
  *yyssp = yystate;

  if (yyss + yystacksize - 1 <= yyssp)
    {
      /* Get the current used size of the three stacks, in elements.  */
      YYSIZE_T yysize = yyssp - yyss + 1;

#ifdef yyoverflow
      {
	/* Give user a chance to reallocate the stack.  Use copies of
	   these so that the &'s don't force the real ones into
	   memory.  */
	YYSTYPE *yyvs1 = yyvs;
	yytype_int16 *yyss1 = yyss;

	/* Each stack pointer address is followed by the size of the
	   data in use in that stack, in bytes.  This used to be a
	   conditional around just the two extra args, but that might
	   be undefined if yyoverflow is a macro.  */
	yyoverflow (YY_("memory exhausted"),
		    &yyss1, yysize * sizeof (*yyssp),
		    &yyvs1, yysize * sizeof (*yyvsp),
		    &yystacksize);

	yyss = yyss1;
	yyvs = yyvs1;
      }
#else /* no yyoverflow */
# ifndef YYSTACK_RELOCATE
      goto yyexhaustedlab;
# else
      /* Extend the stack our own way.  */
      if (YYMAXDEPTH <= yystacksize)
	goto yyexhaustedlab;
      yystacksize *= 2;
      if (YYMAXDEPTH < yystacksize)
	yystacksize = YYMAXDEPTH;

      {
	yytype_int16 *yyss1 = yyss;
	union yyalloc *yyptr =
	  (union yyalloc *) YYSTACK_ALLOC (YYSTACK_BYTES (yystacksize));
	if (! yyptr)
	  goto yyexhaustedlab;
	YYSTACK_RELOCATE (yyss_alloc, yyss);
	YYSTACK_RELOCATE (yyvs_alloc, yyvs);
#  undef YYSTACK_RELOCATE
	if (yyss1 != yyssa)
	  YYSTACK_FREE (yyss1);
      }
# endif
#endif /* no yyoverflow */

      yyssp = yyss + yysize - 1;
      yyvsp = yyvs + yysize - 1;

      YYDPRINTF ((stderr, "Stack size increased to %lu\n",
		  (unsigned long int) yystacksize));

      if (yyss + yystacksize - 1 <= yyssp)
	YYABORT;
    }

  YYDPRINTF ((stderr, "Entering state %d\n", yystate));

  if (yystate == YYFINAL)
    YYACCEPT;

  goto yybackup;

/*-----------.
| yybackup.  |
`-----------*/
yybackup:

  /* Do appropriate processing given the current state.  Read a
     lookahead token if we need one and don't already have one.  */

  /* First try to decide what to do without reference to lookahead token.  */
  yyn = yypact[yystate];
  if (yypact_value_is_default (yyn))
    goto yydefault;

  /* Not known => get a lookahead token if don't already have one.  */

  /* YYCHAR is either YYEMPTY or YYEOF or a valid lookahead symbol.  */
  if (yychar == YYEMPTY)
    {
      YYDPRINTF ((stderr, "Reading a token: "));
      yychar = YYLEX;
    }

  if (yychar <= YYEOF)
    {
      yychar = yytoken = YYEOF;
      YYDPRINTF ((stderr, "Now at end of input.\n"));
    }
  else
    {
      yytoken = YYTRANSLATE (yychar);
      YY_SYMBOL_PRINT ("Next token is", yytoken, &yylval, &yylloc);
    }

  /* If the proper action on seeing token YYTOKEN is to reduce or to
     detect an error, take that action.  */
  yyn += yytoken;
  if (yyn < 0 || YYLAST < yyn || yycheck[yyn] != yytoken)
    goto yydefault;
  yyn = yytable[yyn];
  if (yyn <= 0)
    {
      if (yytable_value_is_error (yyn))
        goto yyerrlab;
      yyn = -yyn;
      goto yyreduce;
    }

  /* Count tokens shifted since error; after three, turn off error
     status.  */
  if (yyerrstatus)
    yyerrstatus--;

  /* Shift the lookahead token.  */
  YY_SYMBOL_PRINT ("Shifting", yytoken, &yylval, &yylloc);

  /* Discard the shifted token.  */
  yychar = YYEMPTY;

  yystate = yyn;
  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  *++yyvsp = yylval;
  YY_IGNORE_MAYBE_UNINITIALIZED_END

  goto yynewstate;


/*-----------------------------------------------------------.
| yydefault -- do the default action for the current state.  |
`-----------------------------------------------------------*/
yydefault:
  yyn = yydefact[yystate];
  if (yyn == 0)
    goto yyerrlab;
  goto yyreduce;


/*-----------------------------.
| yyreduce -- Do a reduction.  |
`-----------------------------*/
yyreduce:
  /* yyn is the number of a rule to reduce with.  */
  yylen = yyr2[yyn];

  /* If YYLEN is nonzero, implement the default value of the action:
     `$$ = $1'.

     Otherwise, the following line sets YYVAL to garbage.
     This behavior is undocumented and Bison
     users should not rely upon it.  Assigning to YYVAL
     unconditionally makes the parser a bit smaller, and it avoids a
     GCC warning that YYVAL may be used uninitialized.  */
  yyval = yyvsp[1-yylen];


  YY_REDUCE_PRINT (yyn);
  switch (yyn)
    {
        case 6:
/* Line 1792 of yacc.c  */
#line 126 "./ldgram.y"
    { ldlex_defsym(); }
    break;

  case 7:
/* Line 1792 of yacc.c  */
#line 128 "./ldgram.y"
    {
		  ldlex_popstate();
		  lang_add_assignment(exp_assop((yyvsp[(3) - (4)].token),(yyvsp[(2) - (4)].name),(yyvsp[(4) - (4)].etree)));
		}
    break;

  case 8:
/* Line 1792 of yacc.c  */
#line 135 "./ldgram.y"
    {    	ldlex_mri_script();
			PUSH_ERROR("MRI style script");
		}
    break;

  case 9:
/* Line 1792 of yacc.c  */
#line 139 "./ldgram.y"
    {	ldlex_popstate(); 
			POP_ERROR();
		}
    break;

  case 14:
/* Line 1792 of yacc.c  */
#line 152 "./ldgram.y"
    {
			einfo("%P%F: unrecognised keyword in MRI style script '%s'\n",(yyvsp[(1) - (1)].name));
			}
    break;

  case 15:
/* Line 1792 of yacc.c  */
#line 155 "./ldgram.y"
    {
			config.map_filename = "-";
			}
    break;

  case 18:
/* Line 1792 of yacc.c  */
#line 161 "./ldgram.y"
    { mri_public((yyvsp[(2) - (4)].name), (yyvsp[(4) - (4)].etree)); }
    break;

  case 19:
/* Line 1792 of yacc.c  */
#line 163 "./ldgram.y"
    { mri_public((yyvsp[(2) - (4)].name), (yyvsp[(4) - (4)].etree)); }
    break;

  case 20:
/* Line 1792 of yacc.c  */
#line 165 "./ldgram.y"
    { mri_public((yyvsp[(2) - (3)].name), (yyvsp[(3) - (3)].etree)); }
    break;

  case 21:
/* Line 1792 of yacc.c  */
#line 167 "./ldgram.y"
    { mri_format((yyvsp[(2) - (2)].name)); }
    break;

  case 22:
/* Line 1792 of yacc.c  */
#line 169 "./ldgram.y"
    { mri_output_section((yyvsp[(2) - (4)].name), (yyvsp[(4) - (4)].etree));}
    break;

  case 23:
/* Line 1792 of yacc.c  */
#line 171 "./ldgram.y"
    { mri_output_section((yyvsp[(2) - (3)].name), (yyvsp[(3) - (3)].etree));}
    break;

  case 24:
/* Line 1792 of yacc.c  */
#line 173 "./ldgram.y"
    { mri_output_section((yyvsp[(2) - (4)].name), (yyvsp[(4) - (4)].etree));}
    break;

  case 25:
/* Line 1792 of yacc.c  */
#line 175 "./ldgram.y"
    { mri_align((yyvsp[(2) - (4)].name),(yyvsp[(4) - (4)].etree)); }
    break;

  case 26:
/* Line 1792 of yacc.c  */
#line 177 "./ldgram.y"
    { mri_alignmod((yyvsp[(2) - (4)].name),(yyvsp[(4) - (4)].etree)); }
    break;

  case 29:
/* Line 1792 of yacc.c  */
#line 181 "./ldgram.y"
    { mri_name((yyvsp[(2) - (2)].name)); }
    break;

  case 30:
/* Line 1792 of yacc.c  */
#line 183 "./ldgram.y"
    { mri_alias((yyvsp[(2) - (4)].name),(yyvsp[(4) - (4)].name),0);}
    break;

  case 31:
/* Line 1792 of yacc.c  */
#line 185 "./ldgram.y"
    { mri_alias((yyvsp[(2) - (4)].name),0,(int) (yyvsp[(4) - (4)].integer));}
    break;

  case 32:
/* Line 1792 of yacc.c  */
#line 187 "./ldgram.y"
    { mri_base((yyvsp[(2) - (2)].etree)); }
    break;

  case 33:
/* Line 1792 of yacc.c  */
#line 189 "./ldgram.y"
    {  mri_truncate((unsigned int) (yyvsp[(2) - (2)].integer)); }
    break;

  case 35:
/* Line 1792 of yacc.c  */
#line 194 "./ldgram.y"
    { mri_order((yyvsp[(3) - (3)].name)); }
    break;

  case 36:
/* Line 1792 of yacc.c  */
#line 195 "./ldgram.y"
    { mri_order((yyvsp[(2) - (2)].name)); }
    break;

  case 38:
/* Line 1792 of yacc.c  */
#line 201 "./ldgram.y"
    { mri_load((yyvsp[(1) - (1)].name)); }
    break;

  case 39:
/* Line 1792 of yacc.c  */
#line 202 "./ldgram.y"
    { mri_load((yyvsp[(3) - (3)].name)); }
    break;

  case 40:
/* Line 1792 of yacc.c  */
#line 207 "./ldgram.y"
    { mri_only_load((yyvsp[(1) - (1)].name)); }
    break;

  case 41:
/* Line 1792 of yacc.c  */
#line 209 "./ldgram.y"
    { mri_only_load((yyvsp[(3) - (3)].name)); }
    break;

  case 42:
/* Line 1792 of yacc.c  */
#line 213 "./ldgram.y"
    {
	 ldlex_both();
	}
    break;

  case 43:
/* Line 1792 of yacc.c  */
#line 217 "./ldgram.y"
    {
	ldlex_popstate();
	}
    break;

  case 54:
/* Line 1792 of yacc.c  */
#line 240 "./ldgram.y"
    { lang_add_target((yyvsp[(3) - (4)].name)); }
    break;

  case 55:
/* Line 1792 of yacc.c  */
#line 242 "./ldgram.y"
    { ldfile_add_library_path ((yyvsp[(3) - (4)].name), false); }
    break;

  case 56:
/* Line 1792 of yacc.c  */
#line 244 "./ldgram.y"
    { lang_add_output((yyvsp[(3) - (4)].name), 1); }
    break;

  case 57:
/* Line 1792 of yacc.c  */
#line 246 "./ldgram.y"
    { lang_add_output_format ((yyvsp[(3) - (4)].name), (char *) NULL,
					    (char *) NULL, 1); }
    break;

  case 58:
/* Line 1792 of yacc.c  */
#line 249 "./ldgram.y"
    { lang_add_output_format ((yyvsp[(3) - (8)].name), (yyvsp[(5) - (8)].name), (yyvsp[(7) - (8)].name), 1); }
    break;

  case 59:
/* Line 1792 of yacc.c  */
#line 251 "./ldgram.y"
    { ldfile_set_output_arch((yyvsp[(3) - (4)].name)); }
    break;

  case 60:
/* Line 1792 of yacc.c  */
#line 253 "./ldgram.y"
    { command_line.force_common_definition = true ; }
    break;

  case 62:
/* Line 1792 of yacc.c  */
#line 256 "./ldgram.y"
    { lang_enter_group (); }
    break;

  case 63:
/* Line 1792 of yacc.c  */
#line 258 "./ldgram.y"
    { lang_leave_group (); }
    break;

  case 64:
/* Line 1792 of yacc.c  */
#line 260 "./ldgram.y"
    { lang_add_map((yyvsp[(3) - (4)].name)); }
    break;

  case 65:
/* Line 1792 of yacc.c  */
#line 262 "./ldgram.y"
    { ldfile_open_command_file((yyvsp[(2) - (2)].name)); }
    break;

  case 67:
/* Line 1792 of yacc.c  */
#line 267 "./ldgram.y"
    { lang_add_input_file((yyvsp[(1) - (1)].name),lang_input_file_is_search_file_enum,
				 (char *)NULL); }
    break;

  case 68:
/* Line 1792 of yacc.c  */
#line 270 "./ldgram.y"
    { lang_add_input_file((yyvsp[(3) - (3)].name),lang_input_file_is_search_file_enum,
				 (char *)NULL); }
    break;

  case 69:
/* Line 1792 of yacc.c  */
#line 273 "./ldgram.y"
    { lang_add_input_file((yyvsp[(2) - (2)].name),lang_input_file_is_search_file_enum,
				 (char *)NULL); }
    break;

  case 74:
/* Line 1792 of yacc.c  */
#line 289 "./ldgram.y"
    { lang_add_entry ((yyvsp[(3) - (4)].name), 0); }
    break;

  case 76:
/* Line 1792 of yacc.c  */
#line 295 "./ldgram.y"
    { lang_add_wild((yyvsp[(1) - (1)].name), current_file); }
    break;

  case 77:
/* Line 1792 of yacc.c  */
#line 297 "./ldgram.y"
    { lang_add_wild((yyvsp[(3) - (3)].name), current_file); }
    break;

  case 78:
/* Line 1792 of yacc.c  */
#line 302 "./ldgram.y"
    {
		lang_add_wild((char *)NULL, (yyvsp[(1) - (1)].name));
		}
    break;

  case 79:
/* Line 1792 of yacc.c  */
#line 306 "./ldgram.y"
    {
			current_file = (char *)NULL;
			}
    break;

  case 81:
/* Line 1792 of yacc.c  */
#line 312 "./ldgram.y"
    {
			current_file =(yyvsp[(1) - (1)].name);
			}
    break;

  case 83:
/* Line 1792 of yacc.c  */
#line 317 "./ldgram.y"
    {
			current_file = (char *)NULL;
			}
    break;

  case 86:
/* Line 1792 of yacc.c  */
#line 326 "./ldgram.y"
    {
 		lang_add_attribute(lang_object_symbols_statement_enum); 
	      	}
    break;

  case 88:
/* Line 1792 of yacc.c  */
#line 331 "./ldgram.y"
    {
 		
		  lang_add_attribute(lang_constructors_statement_enum); 
		}
    break;

  case 90:
/* Line 1792 of yacc.c  */
#line 337 "./ldgram.y"
    {
			lang_add_data((int) (yyvsp[(1) - (4)].integer),(yyvsp[(3) - (4)].etree));
			}
    break;

  case 91:
/* Line 1792 of yacc.c  */
#line 342 "./ldgram.y"
    {
			  lang_add_fill
			    (exp_get_value_int((yyvsp[(3) - (4)].etree),
					       0,
					       "fill value",
					       lang_first_phase_enum));
			}
    break;

  case 96:
/* Line 1792 of yacc.c  */
#line 363 "./ldgram.y"
    { (yyval.integer) = (yyvsp[(1) - (1)].token); }
    break;

  case 97:
/* Line 1792 of yacc.c  */
#line 365 "./ldgram.y"
    { (yyval.integer) = (yyvsp[(1) - (1)].token); }
    break;

  case 98:
/* Line 1792 of yacc.c  */
#line 367 "./ldgram.y"
    { (yyval.integer) = (yyvsp[(1) - (1)].token); }
    break;

  case 99:
/* Line 1792 of yacc.c  */
#line 369 "./ldgram.y"
    { (yyval.integer) = (yyvsp[(1) - (1)].token); }
    break;

  case 100:
/* Line 1792 of yacc.c  */
#line 374 "./ldgram.y"
    {
		  (yyval.integer) =	 exp_get_value_int((yyvsp[(2) - (2)].etree),
					   0,
					   "fill value",
					   lang_first_phase_enum);
		}
    break;

  case 101:
/* Line 1792 of yacc.c  */
#line 380 "./ldgram.y"
    { (yyval.integer) = 0; }
    break;

  case 102:
/* Line 1792 of yacc.c  */
#line 387 "./ldgram.y"
    { (yyval.token) = '+'; }
    break;

  case 103:
/* Line 1792 of yacc.c  */
#line 389 "./ldgram.y"
    { (yyval.token) = '-'; }
    break;

  case 104:
/* Line 1792 of yacc.c  */
#line 391 "./ldgram.y"
    { (yyval.token) = '*'; }
    break;

  case 105:
/* Line 1792 of yacc.c  */
#line 393 "./ldgram.y"
    { (yyval.token) = '/'; }
    break;

  case 106:
/* Line 1792 of yacc.c  */
#line 395 "./ldgram.y"
    { (yyval.token) = LSHIFT; }
    break;

  case 107:
/* Line 1792 of yacc.c  */
#line 397 "./ldgram.y"
    { (yyval.token) = RSHIFT; }
    break;

  case 108:
/* Line 1792 of yacc.c  */
#line 399 "./ldgram.y"
    { (yyval.token) = '&'; }
    break;

  case 109:
/* Line 1792 of yacc.c  */
#line 401 "./ldgram.y"
    { (yyval.token) = '|'; }
    break;

  case 112:
/* Line 1792 of yacc.c  */
#line 411 "./ldgram.y"
    {
		  lang_add_assignment (exp_assop ((yyvsp[(2) - (3)].token), (yyvsp[(1) - (3)].name), (yyvsp[(3) - (3)].etree)));
		}
    break;

  case 113:
/* Line 1792 of yacc.c  */
#line 415 "./ldgram.y"
    {
		  lang_add_assignment (exp_assop ('=', (yyvsp[(1) - (3)].name),
						  exp_binop ((yyvsp[(2) - (3)].token),
							     exp_nameop (NAME,
									 (yyvsp[(1) - (3)].name)),
							     (yyvsp[(3) - (3)].etree))));
		}
    break;

  case 114:
/* Line 1792 of yacc.c  */
#line 423 "./ldgram.y"
    {
		  lang_add_assignment (exp_provide ((yyvsp[(3) - (6)].name), (yyvsp[(5) - (6)].etree)));
		}
    break;

  case 121:
/* Line 1792 of yacc.c  */
#line 445 "./ldgram.y"
    { region = lang_memory_region_lookup((yyvsp[(1) - (1)].name)); }
    break;

  case 123:
/* Line 1792 of yacc.c  */
#line 451 "./ldgram.y"
    { region->current =
		 region->origin =
		 exp_get_vma((yyvsp[(3) - (3)].etree), 0L,"origin", lang_first_phase_enum);
}
    break;

  case 124:
/* Line 1792 of yacc.c  */
#line 457 "./ldgram.y"
    { region->length = exp_get_vma((yyvsp[(3) - (3)].etree),
					       ~((bfd_vma)0),
					       "length",
					       lang_first_phase_enum);
		}
    break;

  case 125:
/* Line 1792 of yacc.c  */
#line 466 "./ldgram.y"
    {
			lang_set_flags(&region->flags, (yyvsp[(2) - (3)].name));
			}
    break;

  case 127:
/* Line 1792 of yacc.c  */
#line 475 "./ldgram.y"
    { lang_startup((yyvsp[(3) - (4)].name)); }
    break;

  case 129:
/* Line 1792 of yacc.c  */
#line 481 "./ldgram.y"
    { ldemul_hll((char *)NULL); }
    break;

  case 130:
/* Line 1792 of yacc.c  */
#line 486 "./ldgram.y"
    { ldemul_hll((yyvsp[(3) - (3)].name)); }
    break;

  case 131:
/* Line 1792 of yacc.c  */
#line 488 "./ldgram.y"
    { ldemul_hll((yyvsp[(1) - (1)].name)); }
    break;

  case 133:
/* Line 1792 of yacc.c  */
#line 496 "./ldgram.y"
    { ldemul_syslib((yyvsp[(3) - (3)].name)); }
    break;

  case 135:
/* Line 1792 of yacc.c  */
#line 502 "./ldgram.y"
    { lang_float(true); }
    break;

  case 136:
/* Line 1792 of yacc.c  */
#line 504 "./ldgram.y"
    { lang_float(false); }
    break;

  case 137:
/* Line 1792 of yacc.c  */
#line 508 "./ldgram.y"
    { ldlex_expression(); }
    break;

  case 138:
/* Line 1792 of yacc.c  */
#line 510 "./ldgram.y"
    { ldlex_popstate(); (yyval.etree)=(yyvsp[(2) - (2)].etree);}
    break;

  case 139:
/* Line 1792 of yacc.c  */
#line 515 "./ldgram.y"
    { (yyval.etree) = exp_unop('-', (yyvsp[(2) - (2)].etree)); }
    break;

  case 140:
/* Line 1792 of yacc.c  */
#line 517 "./ldgram.y"
    { (yyval.etree) = (yyvsp[(2) - (3)].etree); }
    break;

  case 141:
/* Line 1792 of yacc.c  */
#line 519 "./ldgram.y"
    { (yyval.etree) = exp_unop((int) (yyvsp[(1) - (4)].integer),(yyvsp[(3) - (4)].etree)); }
    break;

  case 142:
/* Line 1792 of yacc.c  */
#line 521 "./ldgram.y"
    { (yyval.etree) = exp_unop('!', (yyvsp[(2) - (2)].etree)); }
    break;

  case 143:
/* Line 1792 of yacc.c  */
#line 523 "./ldgram.y"
    { (yyval.etree) = (yyvsp[(2) - (2)].etree); }
    break;

  case 144:
/* Line 1792 of yacc.c  */
#line 525 "./ldgram.y"
    { (yyval.etree) = exp_unop('~', (yyvsp[(2) - (2)].etree));}
    break;

  case 145:
/* Line 1792 of yacc.c  */
#line 528 "./ldgram.y"
    { (yyval.etree) = exp_binop('*', (yyvsp[(1) - (3)].etree), (yyvsp[(3) - (3)].etree)); }
    break;

  case 146:
/* Line 1792 of yacc.c  */
#line 530 "./ldgram.y"
    { (yyval.etree) = exp_binop('/', (yyvsp[(1) - (3)].etree), (yyvsp[(3) - (3)].etree)); }
    break;

  case 147:
/* Line 1792 of yacc.c  */
#line 532 "./ldgram.y"
    { (yyval.etree) = exp_binop('%', (yyvsp[(1) - (3)].etree), (yyvsp[(3) - (3)].etree)); }
    break;

  case 148:
/* Line 1792 of yacc.c  */
#line 534 "./ldgram.y"
    { (yyval.etree) = exp_binop('+', (yyvsp[(1) - (3)].etree), (yyvsp[(3) - (3)].etree)); }
    break;

  case 149:
/* Line 1792 of yacc.c  */
#line 536 "./ldgram.y"
    { (yyval.etree) = exp_binop('-' , (yyvsp[(1) - (3)].etree), (yyvsp[(3) - (3)].etree)); }
    break;

  case 150:
/* Line 1792 of yacc.c  */
#line 538 "./ldgram.y"
    { (yyval.etree) = exp_binop(LSHIFT , (yyvsp[(1) - (3)].etree), (yyvsp[(3) - (3)].etree)); }
    break;

  case 151:
/* Line 1792 of yacc.c  */
#line 540 "./ldgram.y"
    { (yyval.etree) = exp_binop(RSHIFT , (yyvsp[(1) - (3)].etree), (yyvsp[(3) - (3)].etree)); }
    break;

  case 152:
/* Line 1792 of yacc.c  */
#line 542 "./ldgram.y"
    { (yyval.etree) = exp_binop(EQ , (yyvsp[(1) - (3)].etree), (yyvsp[(3) - (3)].etree)); }
    break;

  case 153:
/* Line 1792 of yacc.c  */
#line 544 "./ldgram.y"
    { (yyval.etree) = exp_binop(NE , (yyvsp[(1) - (3)].etree), (yyvsp[(3) - (3)].etree)); }
    break;

  case 154:
/* Line 1792 of yacc.c  */
#line 546 "./ldgram.y"
    { (yyval.etree) = exp_binop(LE , (yyvsp[(1) - (3)].etree), (yyvsp[(3) - (3)].etree)); }
    break;

  case 155:
/* Line 1792 of yacc.c  */
#line 548 "./ldgram.y"
    { (yyval.etree) = exp_binop(GE , (yyvsp[(1) - (3)].etree), (yyvsp[(3) - (3)].etree)); }
    break;

  case 156:
/* Line 1792 of yacc.c  */
#line 550 "./ldgram.y"
    { (yyval.etree) = exp_binop('<' , (yyvsp[(1) - (3)].etree), (yyvsp[(3) - (3)].etree)); }
    break;

  case 157:
/* Line 1792 of yacc.c  */
#line 552 "./ldgram.y"
    { (yyval.etree) = exp_binop('>' , (yyvsp[(1) - (3)].etree), (yyvsp[(3) - (3)].etree)); }
    break;

  case 158:
/* Line 1792 of yacc.c  */
#line 554 "./ldgram.y"
    { (yyval.etree) = exp_binop('&' , (yyvsp[(1) - (3)].etree), (yyvsp[(3) - (3)].etree)); }
    break;

  case 159:
/* Line 1792 of yacc.c  */
#line 556 "./ldgram.y"
    { (yyval.etree) = exp_binop('^' , (yyvsp[(1) - (3)].etree), (yyvsp[(3) - (3)].etree)); }
    break;

  case 160:
/* Line 1792 of yacc.c  */
#line 558 "./ldgram.y"
    { (yyval.etree) = exp_binop('|' , (yyvsp[(1) - (3)].etree), (yyvsp[(3) - (3)].etree)); }
    break;

  case 161:
/* Line 1792 of yacc.c  */
#line 560 "./ldgram.y"
    { (yyval.etree) = exp_trinop('?' , (yyvsp[(1) - (5)].etree), (yyvsp[(3) - (5)].etree), (yyvsp[(5) - (5)].etree)); }
    break;

  case 162:
/* Line 1792 of yacc.c  */
#line 562 "./ldgram.y"
    { (yyval.etree) = exp_binop(ANDAND , (yyvsp[(1) - (3)].etree), (yyvsp[(3) - (3)].etree)); }
    break;

  case 163:
/* Line 1792 of yacc.c  */
#line 564 "./ldgram.y"
    { (yyval.etree) = exp_binop(OROR , (yyvsp[(1) - (3)].etree), (yyvsp[(3) - (3)].etree)); }
    break;

  case 164:
/* Line 1792 of yacc.c  */
#line 566 "./ldgram.y"
    { (yyval.etree) = exp_nameop(DEFINED, (yyvsp[(3) - (4)].name)); }
    break;

  case 165:
/* Line 1792 of yacc.c  */
#line 568 "./ldgram.y"
    { (yyval.etree) = exp_intop((yyvsp[(1) - (1)].integer)); }
    break;

  case 166:
/* Line 1792 of yacc.c  */
#line 570 "./ldgram.y"
    { (yyval.etree) = exp_nameop(SIZEOF_HEADERS,0); }
    break;

  case 167:
/* Line 1792 of yacc.c  */
#line 573 "./ldgram.y"
    { (yyval.etree) = exp_nameop(SIZEOF,(yyvsp[(3) - (4)].name)); }
    break;

  case 168:
/* Line 1792 of yacc.c  */
#line 575 "./ldgram.y"
    { (yyval.etree) = exp_nameop(ADDR,(yyvsp[(3) - (4)].name)); }
    break;

  case 169:
/* Line 1792 of yacc.c  */
#line 577 "./ldgram.y"
    { (yyval.etree) = exp_unop(ABSOLUTE, (yyvsp[(3) - (4)].etree)); }
    break;

  case 170:
/* Line 1792 of yacc.c  */
#line 579 "./ldgram.y"
    { (yyval.etree) = exp_unop(ALIGN_K,(yyvsp[(3) - (4)].etree)); }
    break;

  case 171:
/* Line 1792 of yacc.c  */
#line 581 "./ldgram.y"
    { (yyval.etree) = exp_unop(ALIGN_K,(yyvsp[(3) - (4)].etree)); }
    break;

  case 172:
/* Line 1792 of yacc.c  */
#line 583 "./ldgram.y"
    { (yyval.etree) = exp_nameop(NAME,(yyvsp[(1) - (1)].name)); }
    break;

  case 173:
/* Line 1792 of yacc.c  */
#line 588 "./ldgram.y"
    { (yyval.etree) = (yyvsp[(3) - (4)].etree); }
    break;

  case 174:
/* Line 1792 of yacc.c  */
#line 589 "./ldgram.y"
    { (yyval.etree) = 0; }
    break;

  case 175:
/* Line 1792 of yacc.c  */
#line 592 "./ldgram.y"
    { ldlex_expression(); }
    break;

  case 176:
/* Line 1792 of yacc.c  */
#line 594 "./ldgram.y"
    { ldlex_popstate(); }
    break;

  case 177:
/* Line 1792 of yacc.c  */
#line 596 "./ldgram.y"
    {
			lang_enter_output_section_statement((yyvsp[(1) - (6)].name),(yyvsp[(3) - (6)].etree),typebits,0,0,0,(yyvsp[(4) - (6)].etree));
			}
    break;

  case 178:
/* Line 1792 of yacc.c  */
#line 600 "./ldgram.y"
    {ldlex_expression();}
    break;

  case 179:
/* Line 1792 of yacc.c  */
#line 601 "./ldgram.y"
    {
		  ldlex_popstate();
		  lang_leave_output_section_statement((yyvsp[(12) - (12)].integer), (yyvsp[(11) - (12)].name));
		}
    break;

  case 181:
/* Line 1792 of yacc.c  */
#line 610 "./ldgram.y"
    { typebits = SEC_NEVER_LOAD; }
    break;

  case 182:
/* Line 1792 of yacc.c  */
#line 611 "./ldgram.y"
    { typebits = 0; }
    break;

  case 183:
/* Line 1792 of yacc.c  */
#line 612 "./ldgram.y"
    { typebits = 0; }
    break;

  case 184:
/* Line 1792 of yacc.c  */
#line 613 "./ldgram.y"
    { typebits = 0; }
    break;

  case 185:
/* Line 1792 of yacc.c  */
#line 614 "./ldgram.y"
    { typebits = 0; }
    break;

  case 186:
/* Line 1792 of yacc.c  */
#line 615 "./ldgram.y"
    { typebits = SEC_ALLOC | SEC_LOAD | SEC_HAS_CONTENTS; }
    break;

  case 187:
/* Line 1792 of yacc.c  */
#line 620 "./ldgram.y"
    { (yyval.etree) = (yyvsp[(1) - (2)].etree); typebits =0;}
    break;

  case 188:
/* Line 1792 of yacc.c  */
#line 621 "./ldgram.y"
    { (yyval.etree) = (yyvsp[(1) - (5)].etree); }
    break;

  case 189:
/* Line 1792 of yacc.c  */
#line 622 "./ldgram.y"
    { (yyval.etree)= (etree_type *)NULL; typebits = 0; }
    break;

  case 190:
/* Line 1792 of yacc.c  */
#line 623 "./ldgram.y"
    { (yyval.etree)= (etree_type *)NULL;  }
    break;

  case 191:
/* Line 1792 of yacc.c  */
#line 628 "./ldgram.y"
    { (yyval.name) = (yyvsp[(2) - (2)].name); }
    break;

  case 192:
/* Line 1792 of yacc.c  */
#line 629 "./ldgram.y"
    { (yyval.name) = "*default*"; }
    break;


/* Line 1792 of yacc.c  */
#line 2874 "y.tab.c"
      default: break;
    }
  /* User semantic actions sometimes alter yychar, and that requires
     that yytoken be updated with the new translation.  We take the
     approach of translating immediately before every use of yytoken.
     One alternative is translating here after every semantic action,
     but that translation would be missed if the semantic action invokes
     YYABORT, YYACCEPT, or YYERROR immediately after altering yychar or
     if it invokes YYBACKUP.  In the case of YYABORT or YYACCEPT, an
     incorrect destructor might then be invoked immediately.  In the
     case of YYERROR or YYBACKUP, subsequent parser actions might lead
     to an incorrect destructor call or verbose syntax error message
     before the lookahead is translated.  */
  YY_SYMBOL_PRINT ("-> $$ =", yyr1[yyn], &yyval, &yyloc);

  YYPOPSTACK (yylen);
  yylen = 0;
  YY_STACK_PRINT (yyss, yyssp);

  *++yyvsp = yyval;

  /* Now `shift' the result of the reduction.  Determine what state
     that goes to, based on the state we popped back to and the rule
     number reduced by.  */

  yyn = yyr1[yyn];

  yystate = yypgoto[yyn - YYNTOKENS] + *yyssp;
  if (0 <= yystate && yystate <= YYLAST && yycheck[yystate] == *yyssp)
    yystate = yytable[yystate];
  else
    yystate = yydefgoto[yyn - YYNTOKENS];

  goto yynewstate;


/*------------------------------------.
| yyerrlab -- here on detecting error |
`------------------------------------*/
yyerrlab:
  /* Make sure we have latest lookahead translation.  See comments at
     user semantic actions for why this is necessary.  */
  yytoken = yychar == YYEMPTY ? YYEMPTY : YYTRANSLATE (yychar);

  /* If not already recovering from an error, report this error.  */
  if (!yyerrstatus)
    {
      ++yynerrs;
#if ! YYERROR_VERBOSE
      yyerror (YY_("syntax error"));
#else
# define YYSYNTAX_ERROR yysyntax_error (&yymsg_alloc, &yymsg, \
                                        yyssp, yytoken)
      {
        char const *yymsgp = YY_("syntax error");
        int yysyntax_error_status;
        yysyntax_error_status = YYSYNTAX_ERROR;
        if (yysyntax_error_status == 0)
          yymsgp = yymsg;
        else if (yysyntax_error_status == 1)
          {
            if (yymsg != yymsgbuf)
              YYSTACK_FREE (yymsg);
            yymsg = (char *) YYSTACK_ALLOC (yymsg_alloc);
            if (!yymsg)
              {
                yymsg = yymsgbuf;
                yymsg_alloc = sizeof yymsgbuf;
                yysyntax_error_status = 2;
              }
            else
              {
                yysyntax_error_status = YYSYNTAX_ERROR;
                yymsgp = yymsg;
              }
          }
        yyerror (yymsgp);
        if (yysyntax_error_status == 2)
          goto yyexhaustedlab;
      }
# undef YYSYNTAX_ERROR
#endif
    }



  if (yyerrstatus == 3)
    {
      /* If just tried and failed to reuse lookahead token after an
	 error, discard it.  */

      if (yychar <= YYEOF)
	{
	  /* Return failure if at end of input.  */
	  if (yychar == YYEOF)
	    YYABORT;
	}
      else
	{
	  yydestruct ("Error: discarding",
		      yytoken, &yylval);
	  yychar = YYEMPTY;
	}
    }

  /* Else will try to reuse lookahead token after shifting the error
     token.  */
  goto yyerrlab1;


/*---------------------------------------------------.
| yyerrorlab -- error raised explicitly by YYERROR.  |
`---------------------------------------------------*/
yyerrorlab:

  /* Pacify compilers like GCC when the user code never invokes
     YYERROR and the label yyerrorlab therefore never appears in user
     code.  */
  if (/*CONSTCOND*/ 0)
     goto yyerrorlab;

  /* Do not reclaim the symbols of the rule which action triggered
     this YYERROR.  */
  YYPOPSTACK (yylen);
  yylen = 0;
  YY_STACK_PRINT (yyss, yyssp);
  yystate = *yyssp;
  goto yyerrlab1;


/*-------------------------------------------------------------.
| yyerrlab1 -- common code for both syntax error and YYERROR.  |
`-------------------------------------------------------------*/
yyerrlab1:
  yyerrstatus = 3;	/* Each real token shifted decrements this.  */

  for (;;)
    {
      yyn = yypact[yystate];
      if (!yypact_value_is_default (yyn))
	{
	  yyn += YYTERROR;
	  if (0 <= yyn && yyn <= YYLAST && yycheck[yyn] == YYTERROR)
	    {
	      yyn = yytable[yyn];
	      if (0 < yyn)
		break;
	    }
	}

      /* Pop the current state because it cannot handle the error token.  */
      if (yyssp == yyss)
	YYABORT;


      yydestruct ("Error: popping",
		  yystos[yystate], yyvsp);
      YYPOPSTACK (1);
      yystate = *yyssp;
      YY_STACK_PRINT (yyss, yyssp);
    }

  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  *++yyvsp = yylval;
  YY_IGNORE_MAYBE_UNINITIALIZED_END


  /* Shift the error token.  */
  YY_SYMBOL_PRINT ("Shifting", yystos[yyn], yyvsp, yylsp);

  yystate = yyn;
  goto yynewstate;


/*-------------------------------------.
| yyacceptlab -- YYACCEPT comes here.  |
`-------------------------------------*/
yyacceptlab:
  yyresult = 0;
  goto yyreturn;

/*-----------------------------------.
| yyabortlab -- YYABORT comes here.  |
`-----------------------------------*/
yyabortlab:
  yyresult = 1;
  goto yyreturn;

#if !defined yyoverflow || YYERROR_VERBOSE
/*-------------------------------------------------.
| yyexhaustedlab -- memory exhaustion comes here.  |
`-------------------------------------------------*/
yyexhaustedlab:
  yyerror (YY_("memory exhausted"));
  yyresult = 2;
  /* Fall through.  */
#endif

yyreturn:
  if (yychar != YYEMPTY)
    {
      /* Make sure we have latest lookahead translation.  See comments at
         user semantic actions for why this is necessary.  */
      yytoken = YYTRANSLATE (yychar);
      yydestruct ("Cleanup: discarding lookahead",
                  yytoken, &yylval);
    }
  /* Do not reclaim the symbols of the rule which action triggered
     this YYABORT or YYACCEPT.  */
  YYPOPSTACK (yylen);
  YY_STACK_PRINT (yyss, yyssp);
  while (yyssp != yyss)
    {
      yydestruct ("Cleanup: popping",
		  yystos[*yyssp], yyvsp);
      YYPOPSTACK (1);
    }
#ifndef yyoverflow
  if (yyss != yyssa)
    YYSTACK_FREE (yyss);
#endif
#if YYERROR_VERBOSE
  if (yymsg != yymsgbuf)
    YYSTACK_FREE (yymsg);
#endif
  /* Make sure YYID is used.  */
  return YYID (yyresult);
}


/* Line 2055 of yacc.c  */
#line 631 "./ldgram.y"

void
yyerror(arg) 
     const char *arg;
{ 
  if (ldfile_assumed_script)
    einfo ("%P:%s: file format not recognized; treating as linker script\n",
	   ldfile_input_filename);
  if (error_index > 0 && error_index < ERROR_NAME_MAX)
     einfo ("%P%F:%S: %s in %s\n", arg, error_names[error_index-1]);
  else
     einfo ("%P%F:%S: %s\n", arg);
}
