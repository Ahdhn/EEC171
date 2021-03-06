@echo off
echo Configuring gprof for go32
rem This batch file assumes a unix-type "sed" program

echo # Makefile generated by "configure.bat"> Makefile

if exist config.sed del config.sed

echo "/^###$/ i\				">>config.sed
echo "MY_MACHINE=i386\				">>config.sed
echo "CC=gcc					">>config.sed

echo # >> config.sed

sed -e "s/^\"//" -e "s/\"$//" -e "s/[ 	]*$//" config.sed > config2.sed
sed -f config2.sed Makefile.in >> Makefile
del config.sed
del config2.sed
