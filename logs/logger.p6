use JSON::Tiny;

sub exportData(%data,$filename){
  my $sub_n = "exports\\"~substr($filename,0,*-5)~".json";
  my $dataDir = open $sub_n, :w;
  $dataDir.say(to-json %data);
  $dataDir.close();
};

sub confidenceData(@n) {
  my $av = @n[0];
  my @l = @n[1];
  my @dev = [];
  for 0..(elems @l[0])-1 {
    my $sing_value = +(Str(@l[0][$_].gist)) - $av;
    if $sing_value-$av > 0 {
      @dev.push($sing_value);
    } else {
      @dev.push($sing_value*-1);
    };
  };
  my $init = 0;
  $init += $_ for @dev ;
  my $a = (1-($init/elems @dev))*100 ;
  return [$a] ;
};

sub averageDeviation(@n,$o?) {
  my $init = 0;
  $init += $_ for @n ;
  my $a = $init/elems @n ;
  return [$a,@n];
};

sub numericalLog(@log,$command) {
  my %commandList =
      "average value" => averageDeviation(@log) ,
      "data confidence" => (averageDeviation(@log) ==> confidenceData) ,
      "show me" => "average value or data confidence";
  if %commandList{$command}:exists {
    say $command," is ",%commandList{$command}[0];
    return %commandList{$command}[0];
  }else{
    die "This analysis is not yet coded";
  }
};

sub readLog(%filecontent,$selection,$sub_selection?) {
  my %compiledInfo ;
  if %filecontent{$selection}:exists {
      say "starting logging";
      if %filecontent{$selection}{$sub_selection.gist}:exists {
        my @result = %filecontent{$selection}{$sub_selection};
        for 0..(elems @result[0];) {say(Str(@result[0][$_].gist));};
        say "Amount of results in it: ",elems @result[0] ;
        my $command = prompt "Numerical analysis?\n";
        my $v = numericalLog @result[0], $command ;
        my %compiledInfo =
          "Pure Data" => @result[0],
          "Number of Results" => elems(@result[0]),
          "Result of Analysis" => $v;
        return %compiledInfo ;
      } elsif $sub_selection.contains("show me") {
        say %filecontent{$selection}.keys ;
        return %compiledInfo ;
      };
    } else{
      say %filecontent{$selection};
      return %compiledInfo ;
    };
  };

sub MAIN($filename,$selection,$sub_selection?,$export?){
  my $fileIO = open $filename, :r;
  my $info = $fileIO.slurp;
  my %log = from-json $info;
  my %d = readLog %log, $selection, $sub_selection;
  if $export.contains("export") {
    exportData %d, $filename ;
    say "Done";
  } else {
    say "Done";
  };
};
