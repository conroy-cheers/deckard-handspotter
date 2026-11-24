{
  lib,
  stdenvNoCC,
  python3,
  makeWrapper,
}:
{
  pname,
  description,
  entryPoint,
  src,
  extraInstall ? "",
  tests ? null,
  dependencies,
  runtimeInputs ? [ ],
  environment ? { },
}:
let
  python = python3.withPackages (_: dependencies);
  wrapperArgs =
    lib.optionals (runtimeInputs != [ ]) [
      "--prefix"
      "PATH"
      ":"
      (lib.makeBinPath runtimeInputs)
    ]
    ++ lib.concatLists (
      lib.mapAttrsToList (name: value: [
        "--set"
        name
        value
      ]) environment
    );
in
stdenvNoCC.mkDerivation (finalAttrs: {
  inherit pname src;
  version = "0.1.0";
  nativeBuildInputs = [ makeWrapper ];
  strictDeps = true;
  dontBuild = true;
  installPhase = ''
    runHook preInstall
    dir="$out/libexec/${pname}"
    install -d "$dir"
    cp -R ./. "$dir/"
    ${extraInstall}
    makeWrapper ${python.interpreter} "$out/bin/${pname}" \
      --add-flags "-B $dir/${entryPoint}.py" ${lib.escapeShellArgs wrapperArgs}
    runHook postInstall
  '';
  doInstallCheck = stdenvNoCC.buildPlatform.canExecute stdenvNoCC.hostPlatform;
  installCheckPhase = ''
    runHook preInstallCheck
    "$out/bin/${pname}" --help > /dev/null
    ${lib.optionalString (tests != null) ''
      PYTHONPATH="$out/libexec/${pname}" ${python.interpreter} -B -m unittest discover -s ${tests}
    ''}
    runHook postInstallCheck
  '';
  meta = {
    inherit description;
    mainProgram = finalAttrs.pname;
    license = lib.licenses.mit;
    platforms = lib.platforms.linux;
  };
})
