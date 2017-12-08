#!usr/bin/perl
use utf8;
if(@ARGV<1)
{
	print "Please input one utf8 txt file!\n";
	exit(1);
}
($int)=@ARGV;
open FILE, $int || die "Can't open the $int file!";

while($line=<FILE>)
{
	$line=~s/\r//g;
	$line=~s/\n//g;
	next if $line=~/^\s*$/;

	$line=~s/\xE3\x80\x82/ /g;	#	銆?	$line=~s/\xEF\xBC\x9F/ /g;	#	锛?	$line=~s/\xEF\xBC\x81/ /g;	#	锛?	$line=~s/\xEF\xBC\x8D/ /g;	#	锛?	$line=~s/\xEF\xBC\x82/ /g;	#	锛?	$line=~s/\xE2\x80\x9D/ /g;	#	鈥?	$line=~s/\xEF\xBF\xA5/ /g;	#	锟?	$line=~s/\xEF\xB9\x96/ /g;	#	锕?	$line=~s/\x3F/ /g;	#	?
	$line=~s/\x21/ /g;	#	!
	$line=~s/\x2D/ /g;	#	-
	$line=~s/\x7E/ /g;	#	~
	$line=~s/\x27/ /g;	#	'
	$line=~s/\x22/ /g;	#	"
	$line=~s/\x25/ /g;	#	%
	$line=~s/\x23/ /g;	#	#
	$line=~s/\x40/ /g;	#	@
	$line=~s/\x26/ /g;	#	&
	$line=~s/\x2E/ /g;	#	.
	$line=~s/\x3A/ /g;	#	:
	$line=~s/\x2F/ /g;	#	/
	$line=~s/([0-9])\s*\x3C\s*([0-9])/$1 \xE5\xB0\x8F\xE4\xBA\x8E $2/g;	#	< to 灏忎簬
	$line=~s/([0-9])\s*\x3E\s*([0-9])/$1 \xE5\xA4\xA7\xE4\xBA\x8E $2/g;	#	> to 澶т簬
	$line=~s/\x3C/ /g;	#	<
	$line=~s/\x3E/ /g;	#	>
	$line=~s/[,~`@#\$\^&\*\(\)\[\]_+|\\{}:;\/]/ /g;	
	$line=~s/^\s*//g;
	$line=~s/\s*$//g;
	$line=~s/\s+/ /g;
	print $line."\n";
	
}
close FILE;
