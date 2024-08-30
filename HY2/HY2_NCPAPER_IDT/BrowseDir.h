#ifndef _BROWSERDIR_H_
#define _BROWSERDIR_H_

#ifdef _WIN32 // Windows version
	#include <direct.h>
	#include <io.h>
#else // Linux version
	#include <dirent.h>
	#include <sys/stat.h>	// for mkdir()
#endif
#include <stdio.h> 
#include <fstream>

using namespace std;

class BrowseDir
{
protected:
	//��ų�ʼĿ¼�ľ���·������'\'��β
	char m_szInitDir[260];//[_MAX_PATH];

public:
	//ȱʡ������
	BrowseDir();
	~BrowseDir();

	//���ó�ʼĿ¼Ϊdir���������false����ʾĿ¼������
	bool setInitDir(const char *datasetPath);
	
	//����Ŀ¼dir����filespecָ�����ļ�
	//filespec����ʹ��ͨ��� * ?�����ܰ���·����
	//������Ŀ¼,����ѭ���ķ���
	//�������false,��ʾ��ֹ�����ļ�
	bool browseDir(const char *datasetPath, const char *filespec, vector<string> &actionType);

	bool browseDir(const char *datasetPath, string resultPath, const char *filespec, 
				   vector<string> &actionType, multimap<std::string,std::string> &actionMap);

};

BrowseDir::~BrowseDir()
{
}

#ifdef _WIN32 // Windows version
BrowseDir::BrowseDir()
{
	//�õ�ǰĿ¼��ʼ��m_szInitDir
	getcwd(m_szInitDir,_MAX_PATH);

	//���Ŀ¼�����һ����ĸ����'\',����������һ��'\'
	int len=strlen(m_szInitDir);
	if (m_szInitDir[len-1] != '\\')
		strcat(m_szInitDir,"\\");
}

bool BrowseDir::setInitDir(const char *datasetPath)
{
	//�Ȱ�dirת��Ϊ����·��
	if (_fullpath(m_szInitDir, datasetPath, _MAX_PATH) == NULL)
		return false;

	//�ж�Ŀ¼�Ƿ����
	if (_chdir(m_szInitDir) != 0)
		return false;

	//���Ŀ¼�����һ����ĸ����'\',����������һ��'\'
	int len=strlen(m_szInitDir);
	if (m_szInitDir[len-1] != '\\')
		strcat(m_szInitDir,"\\");

	return true;
}

bool BrowseDir::browseDir(const char *datasetPath, string resultPath, const char *filespec, 
						  vector<string> &actionType, multimap<std::string,std::string> &actionMap)
{
	// ����������ļ��У����½������ļ��зֱ������������ֺ�SVM���
	string dir = resultPath + "features";
	if ( -1 == _access(dir.c_str(), 0) )
		if ( 0 != _mkdir(dir.c_str()) )
			return false;

	dir = resultPath + "quantization";
	if ( -1 == _access(dir.c_str(), 0) )
		if ( 0 != _mkdir(dir.c_str()) )
			return false;

	dir = resultPath + "svm";
	if ( -1 == _access(dir.c_str(), 0) )
		if ( 0 != _mkdir(dir.c_str()) )
			return false;

	_chdir(datasetPath);

    _finddata_t fileinfo;	
    long hFile;
    // �����ļ���·��
	if((hFile = _findfirst("*.*", &fileinfo))==-1)
        cout<<"Not Found!"<<endl;
    else{
        // ����ļ���
		while(_findnext(hFile, &fileinfo)==0){
			if (strcmp(fileinfo.name,".") != 0 && strcmp(fileinfo.name,"..") != 0)
				if (fileinfo.attrib & _A_SUBDIR)	// ֻ��ȡĿ¼����(Ĭ��Ϊ������) 
					if (strcmp(fileinfo.name,"features") != 0 && strcmp(fileinfo.name,"quantization") != 0 && strcmp(fileinfo.name,"svm") != 0)
						actionType.push_back(fileinfo.name);
		}
    }
	vector<string>::iterator it;
	for(it = actionType.begin(); it != actionType.end(); it++)
	{
		string subdir;
		//subdir = m_szInitDir + *it;
		subdir = datasetPath + *it;
		_chdir(subdir.c_str());
		if((hFile = _findfirst("*.*", &fileinfo))!=-1)
		{
			do{
				if (strcmp(fileinfo.name,".") != 0 && strcmp(fileinfo.name,"..") != 0)
				{
					if (!(fileinfo.attrib & _A_SUBDIR))
					{
						string filename = fileinfo.name;
						// ȥ���ļ���׺��
						int index = filename.find_last_of('.');
						string tmpstr = filename.substr(0, index);
						actionMap.insert(pair<std::string, std::string>(*it, tmpstr));
					}
				}//if fileinfo
			}while(_findnext(hFile, &fileinfo)==0);
		}//if hFile
	}// for it
	_findclose(hFile);

	// �������·�������ڣ����½�����·��"features/��ǰ��/"
	for(it = actionType.begin(); it != actionType.end(); it++){		
		string tmpdir = resultPath + "features\\" + *it;
		if ( -1 == _access(tmpdir.c_str(), 0) )
			if ( 0 != _mkdir(tmpdir.c_str()) )
				return false;
	}// for it

	return true;
}


bool BrowseDir::browseDir(const char *datasetPath, const char *filespec, vector<string> &actionType)
{
	_chdir(datasetPath);

    _finddata_t fileinfo;	
    long hFile;
    //�����ļ���·��
	if((hFile = _findfirst("*.*", &fileinfo))==-1)
        cout<<"Not Found!"<<endl;
    else{
        //����ļ���
		while(_findnext(hFile, &fileinfo)==0)
		{
			if (strcmp(fileinfo.name,".") != 0 && strcmp(fileinfo.name,"..") != 0)
				actionType.push_back(fileinfo.name);
		}
    }
    _findclose(hFile);
	return true;
}

#else // Linux version
BrowseDir::BrowseDir()
{
}

bool BrowseDir::browseDir(const char *datasetPath, const char *filespec, vector<string> &actionType)
{
	DIR *dp;
	struct dirent entry;
	struct dirent *entryPtr = NULL;
	dp = opendir(datasetPath);
	if(!dp)
	{
		cout << "Error: wrong directory in '" << datasetPath << "'." << endl;
		return false;
	}
	readdir_r(dp, &entry, &entryPtr);
	while(entryPtr != NULL)
	{
		string strPathName = entry.d_name;
		if(strPathName != "." && strPathName != ".." && entry.d_type == DT_DIR)
		{
			actionType.push_back(strPathName);
			string tmpDataPath = datasetPath + strPathName + "/";
			DIR *subDp;
			struct dirent subEntry;
			struct dirent *subEntryPtr = NULL;
			subDp = opendir(tmpDataPath.c_str());
			if(!subDp)
			{
				cout << "Error: Cannot open directory '" << strPathName << "'." << endl;
				continue;
			}
			closedir(subDp);
		}
		readdir_r(dp, &entry, &entryPtr);	
	}
	closedir(dp);
	return true;
}

bool BrowseDir::browseDir(const char *datasetPath, string resultPath, const char *filespec, 
						  vector<string> &actionType, multimap<std::string,std::string> &actionMap)
{
	// ����������ļ��У��½������ļ��зֱ������������ֺ�SVM���
	string dir = resultPath + "features";
	if ( -1 == access(dir.c_str(), 0) )
		if ( 0 != mkdir(dir.c_str(), 0755) )
			return false;

	dir = resultPath + "quantization";
	if ( -1 == access(dir.c_str(), 0) )
		if ( 0 != mkdir(dir.c_str(), 0755) )
			return false;

	dir = resultPath + "svm";
	if ( -1 == access(dir.c_str(), 0) )
		if ( 0 != mkdir(dir.c_str(), 0755) )
			return false;	

	DIR *dp;
	struct dirent entry;
	struct dirent *entryPtr = NULL;
	dp = opendir(datasetPath);
	if(!dp){
		cout << "Error: wrong directory in '" << datasetPath << "'." << endl;
		return false;
	}
	readdir_r(dp, &entry, &entryPtr);
	while(entryPtr != NULL)
	{
		string strPathName = entry.d_name;
		if(strPathName != "." && strPathName != ".." && entry.d_type == DT_DIR)
		{
			//if (strPathName == "features" || strPathName == "quantization" || strPathName == "svm")
			//	continue;
			actionType.push_back(strPathName);
			string tmpDataPath = datasetPath + strPathName + "/";
			DIR *subDp;
			struct dirent subEntry;
			struct dirent *subEntryPtr = NULL;
			subDp = opendir(tmpDataPath.c_str());
			if(!subDp){
				cout << "Error: Cannot open directory '" << strPathName << "'." << endl;
				continue;
			}
			readdir_r(subDp, &subEntry, &subEntryPtr);
			while(subEntryPtr != NULL)
			{
				string strF = subEntry.d_name;
				if(strF != "." && strF != "..")
				{
					// ȥ���ļ���׺��
					int index = strF.find_last_of('.');
					string tmpstr = strF.substr(0, index);
					actionMap.insert(pair<std::string, std::string>(strPathName, tmpstr));					
				}
				readdir_r(subDp, &subEntry, &subEntryPtr);
			}
			closedir(subDp);
		}
		readdir_r(dp, &entry, &entryPtr);	
	}// while entryPtr
	closedir(dp);

	// �������·�������ڣ����½�����·��"features/��ǰ��/"
	vector<string>::iterator it;
	for(it = actionType.begin(); it != actionType.end(); it++){		
		string tmpdir = resultPath + "features/" + *it;
		if ( -1 == access(tmpdir.c_str(), 0) )
			if ( 0 != mkdir(tmpdir.c_str(), 0755) )
				return false;	
	}// for it

	return true;
}
#endif
#endif