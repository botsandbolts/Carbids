import {spawn} from 'node:child_process';
const at=process.argv.indexOf('--port');
const port=at>=0?process.argv[at+1]:(process.env.PORT||'4173');
const child=spawn(process.env.CARBIDS_PYTHON||'python3',['-m','streamlit','run','app.py','--server.address','0.0.0.0','--server.port',port,'--browser.gatherUsageStats','false'],{stdio:'inherit'});
for(const signal of ['SIGTERM','SIGINT'])process.on(signal,()=>child.kill(signal));
child.on('exit',code=>process.exit(code??1));
