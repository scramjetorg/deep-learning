# Deep Learning - Sequential Model Sequence

This directory demonstrates how to leverage deep learning techniques to recognize audio commands. 


## Requirements

For this Sequence to run properly on your Linux machine use the following command to start <a href="https://docs.scramjet.org/platform/self-hosted-installation/" target="_blank">STH</a>

```bash
$ DEVELOPMENT=true sth --runtime-adapter=process
```
**NOTE:** This Sequence might consume some disk space, clearing out Scramjet Sequence disk space manually might be required from time to time.

```bash
$ sudo rm -r ~/.scramjet_sequences
```

## Install and Run

Install the <a href="https://docs.scramjet.org/platform/self-hosted-installation/" target="_blank">Scramjet Transform Hub </a> (STH) locally or use 
<a href="https://docs.scramjet.org/platform/get-started/" target="_blank">Scramjet's Cloud Platform</a> environment for the Sequence deployment.
For more information on the below commands check the 
<a href="https://docs.scramjet.org/platform/cli-reference/#useful-commands" target="_blank">CLI reference</a> section on Scramjet's Website.

On the Linux terminal execute the following commands:

```bash
# Create a directory __pypackages__ in the same directory as main.py
~/training-script$ mkdir __pypackages__

# Install dependencies in the __pypackages__ folder. 
~/training-script$ pip3 install -t __pypackages__ -r requirements.txt

# Pack the training-script folder into a gzip format
~$ si sequence pack training-script

# Send the training-script.tar.gz Sequence to the Scramjet's Transform-Hub, with a return <Sequence-id> value
~$ si sequence send training-script.tar.gz --progress

# Start the Sequence with arguments
~$ si seq start - --args=[\"aws_key\","\aws_secret\","\aws_bucket\"] # No spacing between args

# Send the audio file as input
~$ si instance input <Instance-id> local/path/to/multi-label-audio.wav -e -t application/octet-stream

# Return list of S3 Bucket objects as output
~$ si instance output <Instance-id>
```

## Audio specifications

Audio wave file required to be sent as input must be one of ten labels:<br/> ['right', 'left', 'no', 'stop', 'down', 'go', 'up', 'yes', 'on', 'off']

### Audio file Details:

Format : Wave<br/>
Duration : <1 second<br/>
Format : PCM<br/>
Format settings : Little / Signed<br/>
Codec ID : 1<br/>
Bit rate mode : Constant<br/>
Bit rate : 256 kb/s<br/>
Channel(s) : 1 channel<br/>
Sampling rate : 16.0 kHz<br/>
Bit depth : 16 bits<br/>



## License

This project is licensed under MIT licenses. 